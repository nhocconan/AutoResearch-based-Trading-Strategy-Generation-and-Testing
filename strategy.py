#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_12h['high'].values
    low_1d = df_12h['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h ATR(14) for position sizing and stops
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]  # First value
    atr_14_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align 1d RSI to 6h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Align 1d ATR to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 12h EMA(34)
        trend_up = price_close > ema_34_12h_aligned[i]
        trend_down = price_close < ema_34_12h_aligned[i]
        
        # Momentum filter: RSI not in extreme territory
        rsi_not_overbought = rsi_14_1d_aligned[i] < 70
        rsi_not_oversold = rsi_14_1d_aligned[i] > 30
        
        # Volatility filter: current volatility not too high
        vol_filter = atr_14_6h[i] < 2.0 * atr_14_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Uptrend + not overbought + volume confirmation + volatility filter
        if trend_up and rsi_not_overbought and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Downtrend + not oversold + volume confirmation + volatility filter
        if trend_down and rsi_not_oversold and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: trend reversal or volatility spike
        exit_long = not trend_up or (atr_14_6h[i] > 2.5 * atr_14_1d_aligned[i])
        exit_short = not trend_down or (atr_14_6h[i] > 2.5 * atr_14_1d_aligned[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Camarilla breakout strategy using 12h EMA trend filter and 1d RSI momentum filter.
# Enters long when price is above 12h EMA(34), RSI < 70, with volume > 1.5x 20-period average.
# Enters short when price is below 12h EMA(34), RSI > 30, with volume > 1.5x 20-period average.
# Uses volatility filter to avoid choppy markets and prevent whipsaws.
# Position size set to 0.25 to manage risk in volatile markets.
# Target: 20-40 trades per year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend with momentum confirmation.