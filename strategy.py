#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Momentum_Conservative"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA34 on close
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d momentum filter: ROC10 on close
    df_1d = get_htf_data(prices, '1d')
    roc10_1d = pd.Series(df_1d['close']).pct_change(10).values
    roc10_1d_aligned = align_htf_to_ltf(prices, df_1d, roc10_1d)
    
    # 1h entry: RSI14 with extreme levels
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(roc10_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema34_val = ema34_4h_aligned[i]
        roc_val = roc10_1d_aligned[i]
        rsi_val = rsi[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: 4h uptrend + 1d positive momentum + RSI oversold bounce
            if close_val > ema34_val and roc_val > 0.02 and rsi_val < 30 and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1d negative momentum + RSI overbought bounce
            elif close_val < ema34_val and roc_val < -0.02 and rsi_val > 70 and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend breaks or RSI overbought
            if close_val < ema34_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend breaks or RSI oversold
            if close_val > ema34_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals