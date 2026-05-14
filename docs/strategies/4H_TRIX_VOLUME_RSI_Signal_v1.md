# Strategy: 4H_TRIX_VOLUME_RSI_Signal_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.203 | +30.2% | -9.9% | 446 | PASS |
| ETHUSDT | 0.021 | +19.2% | -28.3% | 427 | PASS |
| SOLUSDT | 0.444 | +58.5% | -30.0% | 372 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.805 | -2.4% | -9.1% | 144 | FAIL |
| ETHUSDT | 1.172 | +27.0% | -10.9% | 138 | PASS |
| SOLUSDT | 0.380 | +12.4% | -14.9% | 140 | PASS |

## Code
```python
# 4H_TRIX_VOLUME_RSI_Signal_v1
# Hypothesis: TRIX (12-period) momentum combined with volume confirmation and RSI filter
# provides reliable entries in both bull and bear markets. Uses 4h timeframe with 1d RSI
# and volume spike for confirmation. TRIX crossovers signal momentum shifts, filtered
# by volume > 1.5x average and RSI between 30-70 to avoid extremes. Targets 50-150
# trades over 4 years with low frequency to minimize fee drag.

name = "4H_TRIX_VOLUME_RSI_Signal_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- TRIX (12-period) on 4h ---
    # Calculate EMA1, EMA2, EMA3 then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # --- 1d RSI (14-period) ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if RSI is NaN
        if np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                # Simple trailing stop: exit if TRIX reverses
                if position == 1 and trix[i] < trix[i-1]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and trix[i] > trix[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions
        long_entry = (trix[i] > trix[i-1]) and vol_spike[i] and (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        short_entry = (trix[i] < trix[i-1]) and vol_spike[i] and (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on TRIX reversal or RSI extreme
            if position == 1:
                if (trix[i] < trix[i-1]) or (rsi_1d_aligned[i] >= 70) or (rsi_1d_aligned[i] <= 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (trix[i] > trix[i-1]) or (rsi_1d_aligned[i] >= 70) or (rsi_1d_aligned[i] <= 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 05:09
