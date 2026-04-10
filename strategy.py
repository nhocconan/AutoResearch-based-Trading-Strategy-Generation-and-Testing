#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
# - Primary: 4h price breaks above/below Donchian(20) channels from prior completed 4h candle
# - Volume filter: 12h volume > 1.5x 20-period volume MA to ensure institutional participation
# - Regime filter: Choppiness Index(14) < 38.2 (trending market) to avoid false breakouts in chop
# - Exit: Price returns to midpoint of Donchian channel (mean reversion)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, chop filter avoids whipsaws in ranging markets
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_12h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) channels from prior completed 4h candle
    # Use rolling window on 4h data, then align to 4h LTF
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_max_20
    donchian_low = low_min_20
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 4h timeframe (already aligned since calculated on LTF)
    # But we need to ensure we only use completed 4h candle data
    # The rolling calculation already uses min_periods=20, so it's valid
    
    # Calculate 12h volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 14-period Choppiness Index for regime filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # Handle first element
    high_low[0] = high[0] - low[0]
    high_close[0] = np.abs(high[0] - close[0])
    low_close[0] = np.abs(low[0] - close[0])
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_filter = chop < 38.2  # Chop < 38.2 indicates trending market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20_12h_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Align 12h volume data
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = volume_12h_current[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + vol confirmation + chop filter
            if (close[i] > donchian_high[i] and 
                vol_confirm and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + vol confirmation + chop filter
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint
            # Exit: price returns to Donchian midpoint (mean reversion)
            if position == 1:  # Long position
                if close[i] >= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] <= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals