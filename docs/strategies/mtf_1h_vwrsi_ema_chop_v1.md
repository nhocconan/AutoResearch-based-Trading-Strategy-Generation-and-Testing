# Strategy: mtf_1h_vwrsi_ema_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.266 | +0.7% | -6.9% | 261 | FAIL |
| ETHUSDT | 0.584 | +45.0% | -5.9% | 88 | PASS |
| SOLUSDT | 0.251 | +35.2% | -18.6% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.714 | +14.2% | -6.2% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #294: 1h Volume-Weighted RSI + 4h EMA Trend + 1d Choppiness Regime

HYPOTHESIS: Combining volume-weighted RSI on 1h for overextension reversal signals with 4h EMA trend alignment and 1d choppiness regime filter creates a robust mean-reversion strategy that works in both bull and bear markets. The 4h EMA provides medium-term trend direction, the 1d chop filter avoids ranging markets where mean reversion fails, and volume-weighted RSI identifies exhaustion points with institutional participation. Targets 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to minimize fee drag while capturing high-probability reversals at trend extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vwrsi_ema_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for choppiness regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index(14) on 1d data
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1d = np.full(len(close_1d), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 1h timeframe
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators ===
    # Volume-Weighted RSI(14)
    def vwma(series, period):
        """Volume Weighted Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(series, weights, mode='valid') / weights.sum()
    
    # Calculate typical price and volume-weighted changes
    typical_price = (high + low + close) / 3.0
    price_change = np.diff(typical_price, prepend=typical_price[0])
    
    # Separate gains and losses
    gains = np.where(price_change > 0, price_change, 0.0)
    losses = np.where(price_change < 0, -price_change, 0.0)
    
    # Volume-weighted gains and losses
    vol_gains = gains * volume
    vol_losses = losses * volume
    
    # Calculate VW-RSI using Wilder's smoothing (similar to RSI but volume-weighted)
    avg_vol_gain = np.zeros(n)
    avg_vol_loss = np.zeros(n)
    
    # Initialize first values
    if n > 0:
        avg_vol_gain[13] = np.nansum(vol_gains[1:14]) / 14 if np.any(~np.isnan(vol_gains[1:14])) else 0
        avg_vol_loss[13] = np.nansum(vol_losses[1:14]) / 14 if np.any(~np.isnan(vol_losses[1:14])) else 0
    
    # Wilder's smoothing
    for i in range(14, n):
        avg_vol_gain[i] = (avg_vol_gain[i-1] * 13 + vol_gains[i]) / 14
        avg_vol_loss[i] = (avg_vol_loss[i-1] * 13 + vol_losses[i]) / 14
    
    # Calculate VW-RSI
    vwrsi = np.zeros(n)
    for i in range(13, n):
        if avg_vol_loss[i] != 0:
            rs = avg_vol_gain[i] / avg_vol_loss[i]
            vwrsi[i] = 100 - (100 / (1 + rs))
        else:
            vwrsi[i] = 100 if avg_vol_gain[i] > 0 else 50
    
    # For first 13 periods, set to 50 (neutral)
    vwrsi[:13] = 50
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vwrsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid choppy markets (Choppiness > 61.8 = ranging) ---
        # Only trade when market is trending (Choppiness < 38.2) or moderate (38.2-61.8)
        # Avoid strong ranging regimes where mean reversion fails
        if chop_1d_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Price Trend Alignment ---
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # --- Volume-Weighted RSI Signals ---
        # Oversold: VWRSI < 30 (with volume confirmation suggests institutional accumulation)
        # Overbought: VWRSI > 70 (with volume confirmation suggests institutional distribution)
        oversold = vwrsi[i] < 30
        overbought = vwrsi[i] > 70
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at VWRSI > 50 (mean reversion complete)
                if vwrsi[i] > 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at VWRSI < 50 (mean reversion complete)
                if vwrsi[i] < 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold VWRSI with price above 4h EMA (bullish alignment)
        if oversold and price_above_ema:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Overbought VWRSI with price below 4h EMA (bearish alignment)
        elif overbought and price_below_ema:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 09:01
