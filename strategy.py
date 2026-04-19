# 4h_Donchian20_1d_EMA34_Trend_Volume
# Hypothesis: 4h Donchian breakout combined with 1d EMA34 trend filter and volume confirmation reduces false breakouts.
# Works in bull (breakouts with trend) and bear (avoids counter-trend breakouts). Target: 20-50 trades/year.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1d_EMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 with proper Wilder's smoothing (alpha=1/34)
    close_s = pd.Series(close_1d)
    ema_34 = close_s.ewm(alpha=1/34, adjust=False).values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_34_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian + above 1d EMA34 + volume
            if price > upper and price > ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + below 1d EMA34 + volume
            elif price < lower and price < ema and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below 1d EMA34 or Donchian lower band
            if price < ema or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above 1d EMA34 or Donchian upper band
            if price > ema or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals