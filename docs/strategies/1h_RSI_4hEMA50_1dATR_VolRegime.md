# Strategy: 1h_RSI_4hEMA50_1dATR_VolRegime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.200 | +27.8% | -5.6% | 731 | PASS |
| ETHUSDT | 0.644 | +51.6% | -7.9% | 759 | PASS |
| SOLUSDT | 0.459 | +50.6% | -13.2% | 738 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.070 | -0.3% | -6.4% | 249 | FAIL |
| ETHUSDT | 0.654 | +13.0% | -6.9% | 233 | PASS |
| SOLUSDT | 0.231 | +8.8% | -12.2% | 269 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4h trend filter and 1d volatility regime filter.
# Long when: RSI(14) > 55 AND 4h EMA(50) rising AND 1d ATR(14) < 1d ATR(50) (low vol regime)
# Short when: RSI(14) < 45 AND 4h EMA(50) falling AND 1d ATR(14) < 1d ATR(50) (low vol regime)
# Exit when RSI crosses back to 50.
# Designed for 1h timeframe with low trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 4h for trend direction and 1d for volatility regime to avoid choppy markets.
# Works in bull markets via RSI > 55 in uptrend, in bear markets via RSI < 45 in downtrend.
# Volatility filter (ATR14 < ATR50) avoids high-noise periods and whipsaws.
name = "1h_RSI_4hEMA50_1dATR_VolRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_4h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_4h, dtype=bool)
    ema_50_rising[1:] = ema_50_4h[1:] > ema_50_4h[:-1]
    ema_50_falling[1:] = ema_50_4h[1:] < ema_50_4h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_50_falling)
    
    # 1d ATR(14) and ATR(50) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # low volatility regime
    
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 55 AND 4h EMA50 rising AND low vol regime
            long_condition = (rsi[i] > 55) and ema_50_rising_aligned[i] and low_vol_aligned[i]
            # Short: RSI < 45 AND 4h EMA50 falling AND low vol regime
            short_condition = (rsi[i] < 45) and ema_50_falling_aligned[i] and low_vol_aligned[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-05-07 23:17
