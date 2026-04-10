#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d mean reversion with 1w trend filter and volume confirmation
# - Long when RSI(14) < 30 (oversold) AND 1w EMA(20) > EMA(50) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when RSI(14) > 70 (overbought) AND 1w EMA(20) < EMA(50) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when RSI returns to 50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and fee drag
# - RSI captures short-term exhaustion; 1w EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in both bull and bear markets: mean reversion in ranges, trend filter prevents counter-trend trades

name = "1d_1w_rsi_meanreversion_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(20) vs EMA(50)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish_1w = ema_20_1w > ema_50_1w
    ema_bearish_1w = ema_20_1w < ema_50_1w
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = prices['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Align HTF indicators to 1d timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute RSI(14) on 1d data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero (when avg_loss == 0 and avg_gain == 0)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50, rsi)
    
    # RSI conditions: < 30 oversold, > 70 overbought, exit at 50
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    rsi_exit = np.abs(rsi - 50) < 2.5  # Within 2.5 of 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(rsi_oversold[i]) or
            np.isnan(rsi_overbought[i]) or np.isnan(rsi_exit[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when RSI oversold AND 1w bullish trend AND volume spike
            if (rsi_oversold[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short when RSI overbought AND 1w bearish trend AND volume spike
            elif (rsi_overbought[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike_1d[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to RSI = 50 (mean reversion)
            # Exit when RSI returns to equilibrium (50)
            exit_signal = rsi_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals