#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR regime filter
# - Primary: 4h timeframe for optimal balance of trade frequency and fee drag
# - HTF: 1d for volatility (ATR percentile) and volume confirmation
# - Long: Price breaks above 20-period Donchian high + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below 20-period Donchian low + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Exit: ATR-based trailing stop (3x ATR from extreme) or opposite Donchian break
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Works in bull/bear: Donchian captures breakouts in trending markets, volume/ATR filter avoids false signals in ranging markets

name = "4h_1d_donchian_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_entry_price = 0.0
    short_entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid extremely low volatility)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above 20-period Donchian high + vol regime + volume spike
            if (close_4h[i] > highest_high_20[i] and vol_regime and volume_spike):
                position = 1
                long_entry_price = close_4h[i]
                highest_since_long = high_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below 20-period Donchian low + vol regime + volume spike
            elif (close_4h[i] < lowest_low_20[i] and vol_regime and volume_spike):
                position = -1
                short_entry_price = close_4h[i]
                lowest_since_short = low_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update extreme prices for trailing stop
            if position == 1:  # Long position
                highest_since_long = max(highest_since_long, high_4h[i])
                # ATR-based trailing stop: exit if price drops 3x ATR from highest point
                atr_value = atr_1d[i // 16] if i // 16 < len(atr_1d) else atr_1d[-1]
                stop_price = highest_since_long - 3.0 * atr_value
                exit_condition = (close_4h[i] < stop_price)
                
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                lowest_since_short = min(lowest_since_short, low_4h[i])
                # ATR-based trailing stop: exit if price rises 3x ATR from lowest point
                atr_value = atr_1d[i // 16] if i // 16 < len(atr_1d) else atr_1d[-1]
                stop_price = lowest_since_short + 3.0 * atr_value
                exit_condition = (close_4h[i] > stop_price)
                
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals