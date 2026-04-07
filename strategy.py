#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) mean reversion with 1d trend filter and volume confirmation
# Long when Williams %R < -80 (oversold), 1d close > 1d EMA200 (uptrend), and volume > 1.3x 6h average volume
# Short when Williams %R > -20 (overbought), 1d close < 1d EMA200 (downtrend), and volume > 1.3x 6h average volume
# Exit when Williams %R crosses -50 (mean reversion complete) or trend reverses
# Stoploss at 2.5 * ATR(22) to accommodate 6h volatility
# Position size: 0.25 (25% of capital)
# Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_williamsr14_1d_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr = wr.replace([np.inf, -np.inf], np.nan).values
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h volume average for confirmation
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(22) for stoploss (wider for 6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(wr[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_6h[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: WR crosses -50 (mean reversion) or trend reverses (price below EMA200)
            elif wr[i] > -50 or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: WR crosses -50 (mean reversion) or trend reverses (price above EMA200)
            elif wr[i] < -50 or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: WR < -80 (oversold), price above EMA200 (uptrend), volume spike
            if (wr[i] < -80 and
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > 1.3 * volume_ma_6h[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: WR > -20 (overbought), price below EMA200 (downtrend), volume spike
            elif (wr[i] > -20 and
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > 1.3 * volume_ma_6h[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals