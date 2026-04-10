#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volatility regime filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND ATR(14)/ATR(50) > 0.8 (low volatility regime) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND ATR(14)/ATR(50) > 0.8 AND volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian midpoint (mean reversion to equilibrium)
# - Uses ATR ratio to identify low volatility breakouts which work in both bull/bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-50 trades/year on 4h timeframe (120-200 total over 4 years)

name = "4h_1d_donchian_breakout_vol_regime_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute ATR for volatility regime filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR calculations
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio > 0.8 indicates low volatility regime
    
    # Pre-compute Donchian channels from 1d data (using previous day's high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) levels: based on previous 20 days
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    donch_mid = (donch_high_20 + donch_low_20) / 2
    
    # Align HTF levels to LTF
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND low volatility regime AND volume spike
            if (prices['close'].iloc[i] > donch_high_aligned[i] and 
                atr_ratio[i] > 0.8 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND low volatility regime AND volume spike
            elif (prices['close'].iloc[i] < donch_low_aligned[i] and 
                  atr_ratio[i] > 0.8 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price crosses Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donch_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donch_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals