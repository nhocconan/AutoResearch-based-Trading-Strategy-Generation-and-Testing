# 6H_Camarilla_Pivot_R1S1_Breakout_Volume_Filter
# Hypothesis: Camarilla pivot levels (R1, S1) act as intraday support/resistance with strong mean reversion.
# In ranging markets (common in 2025 BTC/ETH), price tends to revert from R1/S1.
# In trending markets, breakouts beyond R1/S1 with volume confirmation indicate strong momentum.
# Uses 1d Camarilla levels for context, volume filter to avoid false breakouts.
# Target: 15-25 trades/year per symbol, avoids overtrading via strict breakout and volume filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # Where C = (H+L+Close)/3 (typical price)
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    r1 = typical_price + hl_range * 1.1 / 12
    s1 = typical_price - hl_range * 1.1 / 12
    r4 = typical_price + hl_range * 1.1 / 2
    s4 = typical_price - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Volume confirmation: 6h volume > 1.5x 24-period (4-day) moving average
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 24  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        
        if position == 0:
            # Long: break above R1 with volume (momentum continuation)
            if price > r1_aligned[i] and vol > 1.5 * vol_ma:
                # Additional filter: avoid buying too close to R4 (overbought)
                if price < r4_aligned[i] * 0.98:  # at least 2% below R4
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Short: break below S1 with volume (momentum continuation)
            elif price < s1_aligned[i] and vol > 1.5 * vol_ma:
                # Additional filter: avoid selling too close to S4 (oversold)
                if price > s4_aligned[i] * 1.02:  # at least 2% above S4
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            # Mean reversion: sell at R1, buy at S1 (when no volume spike)
            elif price > r1_aligned[i] and price < r4_aligned[i]:
                # Sell near resistance if not breaking out with volume
                signals[i] = -0.15
                position = -1
                entry_price = price
            elif price < s1_aligned[i] and price > s4_aligned[i]:
                # Buy near support if not breaking down with volume
                signals[i] = 0.15
                position = 1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below S1 (failed breakout or mean reversion)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit near R4
            elif price > r4_aligned[i] * 0.995:  # 0.5% below R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 (failed breakdown or mean reversion)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit near S4
            elif price < s4_aligned[i] * 1.005:  # 0.5% above S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_Camarilla_Pivot_R1S1_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0