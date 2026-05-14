# Strategy: 6h_WilliamsR_1dEMA50_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.164 | +27.4% | -15.7% | 88 | PASS |
| ETHUSDT | -0.616 | -7.0% | -15.6% | 86 | FAIL |
| SOLUSDT | 0.493 | +57.8% | -24.8% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.637 | +12.4% | -3.8% | 32 | PASS |
| SOLUSDT | 0.003 | +5.3% | -9.1% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA50 trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.3 * 20-period avg volume
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.3 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to manage drawdown (BTC -77% in 2022 → ~19.25% loss at 0.25 exposure)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R is effective in ranging markets and captures reversals, while 1d EMA50 filters for primary trend direction
# Volume confirmation ensures conviction behind moves, reducing false signals
# Works in bull via buying oversold dips in uptrend, works in bear via selling overbought rallies in downtrend

name = "6h_WilliamsR_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Williams %R signals with trend and volume filters
            oversold = williams_r[i] < -80
            overbought = williams_r[i] > -20
            
            # Long: oversold AND uptrend AND volume spike
            if oversold and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: overbought AND downtrend AND volume spike
            elif overbought and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 10:43
