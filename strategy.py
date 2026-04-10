#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND ATR(1d) > 20-period average AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-period Donchian low AND ATR(1d) > 20-period average AND volume > 1.5x 20-bar avg
# - Exit when price returns to Donchian midpoint (mean reversion to equilibrium)
# - Uses 1d ATR for volatility filter to ensure breakouts occur in sufficient volatility environments
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years)
# - Donchian breakouts work in both trending and ranging markets; volatility filter avoids false breakouts in low volatility

name = "4h_1d_donchian_breakout_volume_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 1d ATR 20-period average for volatility filter
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Pre-compute Donchian channels from 1d data (20-period high/low)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2
    
    # Align Donchian levels to LTF
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND sufficient volatility with volume spike
            if (prices['close'].iloc[i] > donch_high_aligned[i] and 
                atr_1d_aligned[i] > atr_ma_20_aligned[i] and  # volatility above average
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND sufficient volatility with volume spike
            elif (prices['close'].iloc[i] < donch_low_aligned[i] and 
                  atr_1d_aligned[i] > atr_ma_20_aligned[i] and  # volatility above average
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
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