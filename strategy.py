# %pip install -q numpy pandas
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# - Bollinger Band width < 50th percentile of 50-period range indicates squeeze (low volatility)
# - Breakout when price crosses upper/lower Bollinger Band with volume > 1.5x 1d average
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Designed to capture volatility expansion after consolidation in both bull and bear markets
# - Target: 20-40 trades/year to avoid excessive fee drag (12h timeframe)

name = "12h_BollingerSqueeze_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band width percentile (50-period lookback for squeeze detection)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
            
        # Squeeze condition: Bollinger Band width < 50th percentile (low volatility)
        squeeze_condition = bb_width_percentile[i] < 50
        
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: squeeze breakout above upper band + uptrend + volume
            if squeeze_condition and close[i] > upper_band[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: squeeze breakout below lower band + downtrend + volume
            elif squeeze_condition and close[i] < lower_band[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle band or trend reverses
            if close[i] < sma.iloc[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle band or trend reverses
            if close[i] > sma.iloc[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals