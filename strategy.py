#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# - Long: Price breaks above 6h Donchian upper channel (20) + price above 1d weekly pivot (bullish bias) + 6h volume > 1.5x 20-period MA
# - Short: Price breaks below 6h Donchian lower channel (20) + price below 1d weekly pivot (bearish bias) + 6h volume > 1.5x 20-period MA
# - Exit: Price returns to 6h Donchian midpoint (mean reversion) OR Donchian width expands > 2x ATR(14) (volatility expansion)
# - Position sizing: 0.25 discrete level
# - Targets ~12-30 trades/year on 6h timeframe. Uses Donchian structure for breakouts,
#   weekly pivot for HTF directional bias (works in bull/bear: pivot adapts to trend),
#   volume confirmation avoids false breakouts, volatility exit captures momentum exhaustion.

name = "6h_1d_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    donchian_width = high_20 - low_20
    
    # Calculate 6h ATR(14) for volatility exit
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d weekly pivot (using prior week's OHLC)
    # Approximate weekly pivot from daily data: (weekly_high + weekly_low + weekly_close)/3
    # Since we don't have weekly data directly, use rolling weekly approximation
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values  # ~1 week
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_14[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        vol_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for Donchian breakouts with pivot bias
            # Long entry: Price breaks above upper channel + above weekly pivot + volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower channel + below weekly pivot + volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to midpoint OR volatility expansion (width > 2x ATR)
            volatility_exit = donchian_width[i] > 2.0 * atr_14[i]
            midpoint_return = False
            
            if position == 1:  # Long position
                midpoint_return = close[i] <= donchian_mid[i]
                if midpoint_return or volatility_exit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                midpoint_return = close[i] >= donchian_mid[i]
                if midpoint_return or volatility_exit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals