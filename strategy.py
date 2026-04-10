#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(14) mean reversion + Bollinger Bands chop filter
# - Long when KAMA(10,2,30) rising AND RSI(14) < 30 AND price < BB lower band (20,2.0)
# - Short when KAMA(10,2,30) falling AND RSI(14) > 70 AND price > BB upper band (20,2.0)
# - Exit when RSI crosses 50 (mean reversion completion) or opposite signal
# - Uses 1w HTF trend filter: only take longs when price > 1w EMA(50), shorts when price < 1w EMA(50)
# - Discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in bull markets via trend-following KAMA, in bear markets via RSI mean reversion at extremes

name = "1d_1w_kama_rsi_bb_v1"
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
    
    # Pre-compute 1d KAMA trend direction
    close = prices['close'].values
    # Efficiency ratio ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Pad volatility array for ER calculation
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility[10:]])
    er = change / volatility_padded[10:]
    er = np.concatenate([np.full(10, np.nan), er])
    # Smoothing constants: fastest SC=2/(2+1)=0.6667, slowest SC=2/(30+1)=0.0645
    sc = (er * 0.602 + 0.0645) ** 2  # SC = [ER*(fastest-slowest) + slowest]^2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # Pre-compute 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    rsi_exit = (rsi > 50) & (np.roll(rsi, 1) <= 50)  # RSI crossing above 50
    rsi_exit_short = (rsi < 50) & (np.roll(rsi, 1) >= 50)  # RSI crossing below 50
    
    # Pre-compute 1d Bollinger Bands (20,2.0)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_squeeze = (bb_upper - bb_lower) / bb_middle < 0.1  # Chopper regime when bands narrow
    price_below_lower = close < bb_lower
    price_above_upper = close > bb_upper
    
    # Pre-compute 1w EMA(50) trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    price_above_1w_ema = close > ema_50_1w[-1] if len(ema_50_1w) > 0 else False  # Simplified for alignment
    
    # Align HTF indicators to 1d timeframe
    # For 1w EMA, we need to align properly
    ema_50_1w_series = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_1w_values = ema_50_1w_series.values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_values)
    price_above_1w_ema_aligned = close > ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: KAMA rising AND RSI oversold AND price below lower BB
            if (kama_rising[i] and rsi_oversold[i] and price_below_lower[i] and 
                price_above_1w_ema_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: KAMA falling AND RSI overbought AND price above upper BB
            elif (kama_falling[i] and rsi_overbought[i] and price_above_upper[i] and 
                  not price_above_1w_ema_aligned[i]):  # price below 1w EMA for short
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when RSI crosses above 50 or opposite signal
            exit_long = (position == 1 and 
                       (rsi_exit[i] or 
                        (kama_falling[i] and rsi > 50 and price_above_upper[i])) or
                        not price_above_1w_ema_aligned[i])
            # Exit short when RSI crosses below 50 or opposite signal
            exit_short = (position == -1 and 
                         (rsi_exit_short[i] or 
                          (kama_rising[i] and rsi < 50 and price_below_lower[i]) or
                          price_above_1w_ema_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals