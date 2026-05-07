#!/usr/bin/env python3
name = "6h_Contrarian_RSI_With_Volume_Regime"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d RSI with proper calculation
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rs = rs.replace(0, 1e-10)
    rsi1d = 100 - (100 / (1 + rs))
    rsi1d = rsi1d.fillna(50).values
    
    # 1d Close for trend context
    close1d = df_1d['close'].values
    
    # Align 1d data to 6h
    rsi1d_aligned = align_htf_to_ltf(prices, df_1d, rsi1d)
    close1d_aligned = align_htf_to_ltf(prices, df_1d, close1d)
    
    # 6h RSI for overbought/oversold
    delta6 = pd.Series(close).diff()
    gain6 = (delta6.where(delta6 > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss6 = (-delta6.where(delta6 < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs6 = gain6 / loss6
    rs6 = rs6.replace(0, 1e-10)
    rsi6 = 100 - (100 / (1 + rs6))
    rsi6 = rsi6.fillna(50).values
    
    # 6h volume regime: high volume when above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if np.isnan(rsi1d_aligned[i]) or np.isnan(close1d_aligned[i]) or np.isnan(rsi6[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Contrarian long: Oversold on 6h RSI, but 1d trend is up (close > previous close)
            if rsi6[i] < 30 and close1d_aligned[i] > close1d_aligned[i-1] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Contrarian short: Overbought on 6h RSI, but 1d trend is down
            elif rsi6[i] > 70 and close1d_aligned[i] < close1d_aligned[i-1] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if rsi6[i] > 50 or close1d_aligned[i] < close1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if rsi6[i] < 50 or close1d_aligned[i] > close1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Contrarian mean reversion on 6s timeframe with 1d trend filter and volume confirmation.
# In bull markets: buy 6s oversold (RSI<30) when 1d trend is up, sell when RSI>50 or trend turns down.
# In bear markets: sell 6s overbought (RSI>70) when 1d trend is down, buy when RSI<50 or trend turns up.
# Volume regime filter ensures trades occur during active participation, reducing false signals.
# Uses discrete 0.25 position sizing to limit risk and reduce fee churn.
# Target: 20-40 trades/year to avoid overtrading while capturing mean reversion opportunities.