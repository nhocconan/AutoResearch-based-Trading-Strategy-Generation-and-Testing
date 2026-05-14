# Strategy: 4h_BB_Squeeze_1dEMA200_VolumeConfirm_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.186 | +26.7% | -6.5% | 94 | PASS |
| ETHUSDT | -0.221 | +13.6% | -7.9% | 87 | FAIL |
| SOLUSDT | 0.440 | +43.5% | -10.2% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.453 | +9.1% | -3.4% | 35 | PASS |
| SOLUSDT | -0.294 | +2.6% | -8.5% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above upper BB after low volatility (BB width < 20th percentile) AND price > 1d EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below lower BB after low volatility (BB width < 20th percentile) AND price < 1d EMA200 AND volume > 1.5x 20-period average
# ATR trailing stop (2.0x ATR) to manage risk
# Bollinger squeeze identifies low volatility breakouts, EMA200 filters for higher-timeframe trend, volume confirms conviction
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA200 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h Bollinger Bands (20, 2) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate BB
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Align HTF data
    sma_20_aligned = align_htf_to_ltf(prices, df_4h, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_4h, bb_width)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile_20 = pd.Series(bb_width).expanding(min_periods=20).quantile(0.20).values
    bb_width_percentile_20_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile_20)
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === 4h ATR for trailing stop (14-period) ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(bb_width_percentile_20_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        sma_20_val = sma_20_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        bb_width_val = bb_width_aligned[i]
        bb_width_percentile_20_val = bb_width_percentile_20_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        atr_val = atr_aligned[i]
        
        # Squeeze condition: low volatility (BB width < 20th percentile)
        squeeze_condition = bb_width_val < bb_width_percentile_20_val
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (BB middle touch) ===
        if position == 1:  # Long position
            # Exit when price touches or crosses below BB middle (SMA20)
            if price <= sma_20_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses above BB middle (SMA20)
            if price >= sma_20_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper BB AND squeeze condition AND price > 1d EMA200 AND volume confirmation
            if price > upper_bb_val and squeeze_condition and price > ema_200_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below lower BB AND squeeze condition AND price < 1d EMA200 AND volume confirmation
            elif price < lower_bb_val and squeeze_condition and price < ema_200_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BB_Squeeze_1dEMA200_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 22:21
