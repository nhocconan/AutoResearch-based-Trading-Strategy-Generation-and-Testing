#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 14-period RSI with 1-day Bollinger Band regime filter
# Long when RSI(14) < 30 + price < BB lower (oversold bounce) + 1-day BB width > 50th percentile (volatile regime)
# Short when RSI(14) > 70 + price > BB upper (overbought reversal) + 1-day BB width > 50th percentile
# Exit when RSI crosses 50 (mean reversion complete) or BB width < 30th percentile (low volatility)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Bollinger Bands for regime and volatility filtering
# Target: 80-160 total trades over 4 years (20-40/year)

name = "6h_rsi14_1d_bb_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Bollinger Bands and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter (50th percentile = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Align BB levels to 6h
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # 6-period RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: RSI crosses 50 (mean reversion) or low volatility regime
            elif rsi[i] >= 50 or bb_width_percentile_aligned[i] < 30:
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
            # Exit: RSI crosses 50 (mean reversion) or low volatility regime
            elif rsi[i] <= 50 or bb_width_percentile_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extremes with BB bands and volatility regime
            # Volatility filter: BB width > 50th percentile (volatile enough for mean reversion)
            volatile_regime = bb_width_percentile_aligned[i] > 50
            
            # Long: RSI oversold + price below BB lower + volatile regime
            if rsi[i] < 30 and close[i] < bb_lower_aligned[i] and volatile_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: RSI overbought + price above BB upper + volatile regime
            elif rsi[i] > 70 and close[i] > bb_upper_aligned[i] and volatile_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals