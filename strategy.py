#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation
# - Long when RSI(2) < 10 AND 4h close > 4h EMA(50) AND volume > 1.5x 20-period average
# - Short when RSI(2) > 90 AND 4h close < 4h EMA(50) AND volume > 1.5x 20-period average
# - Exit when RSI(2) crosses above 50 (for longs) or below 50 (for shorts)
# - RSI(2) captures extreme short-term reversals in both bull and bear markets
# - 4h EMA(50) filter ensures we trade with higher timeframe trend (avoids counter-trend whipsaws)
# - Volume confirmation prevents low-participation false signals
# - Target: 15-37 trades/year on 1h (60-150 total over 4 years) to avoid fee drag
# - Discrete position sizing: 0.20 (20% of capital) to control drawdown

name = "1h_rsi2_meanrev_4htrend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Pre-compute 1h RSI(2)
    close_1h = prices['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    
    if len(gain) >= 2:
        avg_gain[1] = np.mean(gain[0:2])
        avg_loss[1] = np.mean(loss[0:2])
        
        for i in range(2, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.full_like(avg_gain, np.nan, dtype=float)
    mask = ~np.isnan(avg_loss) & (avg_loss != 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi_2 = np.full_like(rs, np.nan, dtype=float)
    rsi_2 = 100 - (100 / (1 + rs))
    # Handle edge cases
    rsi_2[avg_loss == 0] = 100
    rsi_2[avg_gain == 0] = 0
    
    # Pre-compute 1h volume MA(20)
    vol_1h = prices['volume'].values
    vol_ma_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1h > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi_2[i]) or np.isnan(ema_4h_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        rsi_now = rsi_2[i]
        close_1h_now = close_1h[i]
        ema_4h_now = ema_4h_50_aligned[i]
        vol_spike_now = vol_spike[i]
        
        # RSI(2) extremes
        rsi_oversold = rsi_now < 10
        rsi_overbought = rsi_now > 90
        rsi_exit_long = rsi_now > 50  # Exit long when RSI > 50
        rsi_exit_short = rsi_now < 50  # Exit short when RSI < 50
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: RSI oversold AND 4h uptrend (price > EMA50) AND volume spike
            if (rsi_oversold and close_1h_now > ema_4h_now and vol_spike_now):
                position = 1
                signals[i] = 0.20
            # Short conditions: RSI overbought AND 4h downtrend (price < EMA50) AND volume spike
            elif (rsi_overbought and close_1h_now < ema_4h_now and vol_spike_now):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: RSI crosses 50 (mean reversion complete)
            exit_long = (position == 1 and rsi_exit_long)
            exit_short = (position == -1 and rsi_exit_short)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals