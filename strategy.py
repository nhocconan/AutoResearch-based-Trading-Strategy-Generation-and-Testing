#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Primary: 6h price breaking above R4 or below S4 Camarilla levels from prior 1d
# - HTF trend filter: 1w close > 1w EMA50 for long bias, < EMA50 for short bias
# - HTF volume filter: 1d volume > 1.3x 20-period MA for institutional participation
# - Entry: Long when close > R4 + volume filter + 1w uptrend; Short when close < S4 + volume filter + 1w downtrend
# - Exit: Price retouches the pivot point (PP) or opposite Camarilla level (R3/S3)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms validity, 1w trend ensures alignment with higher timeframe momentum

name = "6h_1d_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from prior 1d (lookback period)
    def calculate_camarilla(high, low, close):
        typical_price = (high + low + close) / 3.0
        range_hl = high - low
        pp = typical_price
        r1 = pp + range_hl * 1.1 / 12
        r2 = pp + range_hl * 1.1 / 6
        r3 = pp + range_hl * 1.1 / 4
        r4 = pp + range_hl * 1.1 / 2
        s1 = pp - range_hl * 1.1 / 12
        s2 = pp - range_hl * 1.1 / 6
        s3 = pp - range_hl * 1.1 / 4
        s4 = pp - range_hl * 1.1 / 2
        return pp, r1, r2, r3, r4, s1, s2, s3, s4
    
    # Calculate Camarilla on prior 1d (shifted by 1 to avoid look-ahead)
    pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    # Shift by 1 to use only prior completed 1d bar for level calculation
    pp = np.concatenate([np.full(24, np.nan), pp[:-1]])  # 24 * 6h = 1d
    r1 = np.concatenate([np.full(24, np.nan), r1[:-1]])
    r2 = np.concatenate([np.full(24, np.nan), r2[:-1]])
    r3 = np.concatenate([np.full(24, np.nan), r3[:-1]])
    r4 = np.concatenate([np.full(24, np.nan), r4[:-1]])
    s1 = np.concatenate([np.full(24, np.nan), s1[:-1]])
    s2 = np.concatenate([np.full(24, np.nan), s2[:-1]])
    s3 = np.concatenate([np.full(24, np.nan), s3[:-1]])
    s4 = np.concatenate([np.full(24, np.nan), s4[:-1]])
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(pp[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: 1w close > EMA50 for uptrend, < EMA50 for downtrend
        trend_up = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False
        trend_down = close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > R4 + volume confirmation + 1w uptrend
            if (close[i] > r4[i] and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: close < S4 + volume confirmation + 1w downtrend
            elif (close[i] < s4[i] and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price retouches pivot point (PP) OR touches R3/S3
            if position == 1:  # Long position
                if close[i] <= pp[i] or close[i] >= r3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= pp[i] or close[i] <= s3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals