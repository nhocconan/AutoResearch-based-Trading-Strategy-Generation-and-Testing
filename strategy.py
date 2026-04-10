#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and trend filter
# - Primary: 1d price breaking above/below 20-period Donchian channels from prior day
# - HTF volume filter: 1w volume > 2.0x 10-period MA for institutional participation
# - HTF trend filter: 1w close > 1w EMA30 for long bias, < EMA30 for short bias
# - Entry: Long when close > upper Donchian + volume filter + 1w uptrend
#          Short when close < lower Donchian + volume filter + 1w downtrend
# - Exit: Price retouches the middle of the Donchian channel (median of upper/lower)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# - Works in bull/bear: Donchian adapts to volatility, volume confirms validity, 1w trend ensures alignment

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute HTF data
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period) from prior 1d session
    # Using rolling window with min_periods to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donchian = high_20
    lower_donchian = low_20
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Calculate 1w volume MA(10)
    volume_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_10_1w)
    
    # Calculate 1w EMA(30) for trend filter
    ema_30_1w = pd.Series(close_1w).ewm(span=30, min_periods=30, adjust=False).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(40, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(middle_donchian[i]) or np.isnan(volume_ma_10_1w_aligned[i]) or
            np.isnan(ema_30_1w_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2.0x 10-period MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirm = volume_1w_aligned[i] > 2.0 * volume_ma_10_1w_aligned[i]
        
        # Trend filter: 1w close > EMA30 for uptrend, < EMA30 for downtrend
        trend_up = close_1w[-1] > ema_30_1w[-1] if len(close_1w) > 0 else False
        trend_down = close_1w[-1] < ema_30_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > upper Donchian + volume confirmation + 1w uptrend
            if (close[i] > upper_donchian[i] and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: close < lower Donchian + volume confirmation + 1w downtrend
            elif (close[i] < lower_donchian[i] and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price retouches middle of Donchian channel
            if position == 1:  # Long position
                if close[i] <= middle_donchian[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= middle_donchian[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals