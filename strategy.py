#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Primary: 12h price breaking above/below Camarilla H3/L3 levels from prior 1d session
# - HTF volume filter: 1d volume > 1.8x 20-period MA for institutional participation
# - HTF trend filter: 1w close > 1w EMA20 for long bias, < EMA20 for short bias
# - Entry: Long when close > H3 + volume filter + 1w uptrend; Short when close < L3 + volume filter + 1w downtrend
# - Exit: Price retouches Camarilla pivot point (PP) from prior 1d session
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms validity, 1w trend ensures alignment with higher timeframe momentum

name = "12h_1d_1w_camarilla_pivot_trend_v1"
timeframe = "12h"
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from prior 1d session
    def calculate_camarilla(high, low, close):
        # Camarilla equations for H3, L3, and PP (pivot point)
        range_val = high - low
        pp = (high + low + close) / 3.0
        h3 = pp + (range_val * 1.1 / 4.0)
        l3 = pp - (range_val * 1.1 / 4.0)
        return h3, l3, pp
    
    h3_1d, l3_1d, pp_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: 1w close > EMA20 for uptrend, < EMA20 for downtrend
        # Use the latest completed 1w bar (already aligned)
        trend_up = close_1w[-1] > ema_20_1w[-1] if len(close_1w) > 0 else False
        trend_down = close_1w[-1] < ema_20_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > H3 + volume confirmation + 1w uptrend
            if (close[i] > h3_1d_aligned[i] and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: close < L3 + volume confirmation + 1w downtrend
            elif (close[i] < l3_1d_aligned[i] and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price retouches Camarilla pivot point (PP) from prior 1d session
            if position == 1:  # Long position
                if close[i] <= pp_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= pp_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals