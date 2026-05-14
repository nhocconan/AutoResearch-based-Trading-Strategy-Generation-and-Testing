# Strategy: 4h_Donchian20_12hEMA34_Volume_ATRRegime_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.026 | +21.3% | -21.5% | 230 | PASS |
| ETHUSDT | 1.018 | +93.1% | -9.2% | 218 | PASS |
| SOLUSDT | 0.470 | +64.3% | -18.2% | 211 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.074 | -11.1% | -13.1% | 99 | FAIL |
| ETHUSDT | 0.395 | +11.7% | -9.5% | 81 | PASS |
| SOLUSDT | 0.004 | +5.3% | -11.0% | 77 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA34) and volume confirmation.
# Long when price breaks above Donchian upper (20-period high) AND close > 12h EMA34 AND volume > 1.8x 20-period average.
# Short when price breaks below Donchian lower (20-period low) AND close < 12h EMA34 AND volume > 1.8x 20-period average.
# Exit when price returns to Donchian midpoint or ATR(10) < ATR(30) (contracting volatility).
# Uses discrete position size 0.30. 12h EMA34 provides smoother trend filter than 1d EMA50, reducing whipsaw.
# Volume threshold increased to 1.8x to reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 4h data for Donchian channels, volume, and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period) on 4h
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # True Range for ATR calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) and ATR(30) for regime filter
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Align ATR values to 4h timeframe
    atr_10_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    atr_30_aligned = align_htf_to_ltf(prices, df_4h, atr_30)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or np.isnan(atr_30_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_34_val = ema_34_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        midpoint_val = midpoint_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_10_val = atr_10_aligned[i]
        atr_30_val = atr_30_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint or ATR contracts
            if price <= midpoint_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint or ATR contracts
            if price >= midpoint_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.8x 20-period average (stricter to reduce trades)
            vol_filter = vol > 1.8 * vol_ma_val
            
            # Trend filter: price relative to 12h EMA34
            trend_filter_long = price > ema_34_val
            trend_filter_short = price < ema_34_val
            
            # LONG: price breaks above Donchian upper with volume and trend confirmation
            if price > upper_val and vol_filter and trend_filter_long:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume and trend confirmation
            elif price < lower_val and vol_filter and trend_filter_short:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume_ATRRegime_V1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 06:07
