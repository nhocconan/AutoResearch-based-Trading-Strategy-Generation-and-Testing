# Strategy: 4h_Chop_Donchian_MeanRev_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.147 | +12.2% | -14.1% | 66 | FAIL |
| ETHUSDT | 0.338 | +40.9% | -11.0% | 52 | PASS |
| SOLUSDT | 0.940 | +159.2% | -23.3% | 56 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.270 | +9.6% | -7.1% | 13 | PASS |
| SOLUSDT | -0.384 | +0.2% | -8.5% | 11 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter combined with 1d Donchian breakout.
# Uses Choppiness Index (14) on 4h to detect ranging (choppy > 61.8) vs trending (choppy < 38.2) markets.
# In ranging markets: mean reversion at Donchian bands (sell near upper band, buy near lower band).
# In trending markets: breakout continuation (buy above upper band, sell below lower band).
# Volume confirmation required (>1.5x average). Position size 0.25 to manage drawdown.
# Designed to work in both bull (trend breakouts) and bear (mean reversion in ranges).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for Donchian levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h ATR(14) for volatility and stoploss ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # === 4h Choppiness Index (14) ===
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    atr_1 = np.maximum(high_4h - low_4h, 
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    atr_1[0] = 0
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum()
    atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).mean()
    chop = 100 * (np.log10(sum_atr_1) - np.log10(14 * atr_14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop_values)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    
    # === 1d Donchian(20) for breakout/mean reversion levels ===
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(vol_ratio_4h[i]) or
            np.isnan(donch_high_1d_aligned[i]) or
            np.isnan(donch_low_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        atr = atr_14_4h_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        donch_high = donch_high_1d_aligned[i]
        donch_low = donch_low_1d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit conditions depend on regime
            if chop_val > 61.8:  # Ranging market: mean reversion
                if price >= donch_high:  # Hit upper band, take profit
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            else:  # Trending market: trend continuation or reversal
                if price <= donch_low or price < (donch_high + donch_low) / 2:  # Breakdown or midpoint
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        elif position == -1:  # Short position
            # Exit conditions depend on regime
            if chop_val > 61.8:  # Ranging market: mean reversion
                if price <= donch_low:  # Hit lower band, take profit
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            else:  # Trending market: trend continuation or reversal
                if price >= donch_high or price > (donch_high + donch_low) / 2:  # Breakout or midpoint
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if chop_val > 61.8:  # Ranging market: mean reversion
                # LONG: price near lower Donchian band with volume
                if price <= donch_low * 1.005 and vol_ratio > 1.5:  # Within 0.5% of lower band
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: price near upper Donchian band with volume
                elif price >= donch_high * 0.995 and vol_ratio > 1.5:  # Within 0.5% of upper band
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            else:  # Trending market: breakout continuation
                # LONG: break above upper Donchian band with volume
                if price > donch_high and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: break below lower Donchian band with volume
                elif price < donch_low and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_MeanRev_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 19:39
