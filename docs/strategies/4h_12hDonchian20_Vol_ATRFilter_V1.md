# Strategy: 4h_12hDonchian20_Vol_ATRFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.067 | +16.2% | -16.4% | 62 | FAIL |
| ETHUSDT | 0.311 | +38.4% | -19.9% | 52 | PASS |
| SOLUSDT | 0.324 | +39.2% | -21.5% | 26 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.349 | +11.2% | -8.6% | 21 | PASS |
| SOLUSDT | 0.608 | +17.3% | -8.5% | 13 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ATR stoploss.
# Long when price breaks above 12h Donchian upper channel (20) with volume > 1.5x 20-period average AND ATR(14) < 0.025 * price.
# Short when price breaks below 12h Donchian lower channel (20) with volume > 1.5x 20-period average AND ATR(14) < 0.025 * price.
# Exit when price reaches opposite Donchian channel (mean reversion) or ATR-based stoploss (2 * ATR).
# Uses discrete position size 0.25. 12h Donchian provides structure, 4h provides entry timing and volatility filter.
# Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 12h Indicators: Donchian Channels (20) ===
    # Upper channel = highest high over 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel = lowest low over 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to primary timeframe (4h)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # === 4h Indicators: ATR (14) for volatility filter and stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (alpha = 1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        atr = atr_14[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reaches lower Donchian channel (mean reversion) or ATR stoploss
            if price <= lower or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reaches upper Donchian channel (mean reversion) or ATR stoploss
            if price >= upper or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility filter: only trade when ATR < 2.5% of price (low volatility environment)
            vol_filter = atr < 0.025 * price
            
            # LONG: Price breaks above upper Donchian with volume confirmation and low volatility
            if (price > upper) and (vol > 1.5 * vol_ma) and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower Donchian with volume confirmation and low volatility
            elif (price < lower) and (vol > 1.5 * vol_ma) and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_12hDonchian20_Vol_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 05:32
