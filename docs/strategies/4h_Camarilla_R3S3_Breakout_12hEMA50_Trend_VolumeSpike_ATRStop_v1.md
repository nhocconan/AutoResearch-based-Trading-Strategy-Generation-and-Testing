# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.128 | +26.2% | -12.0% | 211 | PASS |
| ETHUSDT | 0.631 | +72.2% | -16.9% | 194 | PASS |
| SOLUSDT | 0.561 | +83.1% | -30.0% | 160 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.304 | +2.0% | -9.4% | 70 | FAIL |
| ETHUSDT | 1.492 | +37.8% | -9.3% | 68 | PASS |
| SOLUSDT | 0.702 | +19.5% | -13.1% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume spike confirmation, and ATR-based trailing stop.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter (price above/below EMA).
- Entry: Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR(14) trailing stop (long: highest_high - 2.5*ATR; short: lowest_low + 2.5*ATR) or opposite Camarilla breakout.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Designed to capture strong trends with volatility-adjusted exits and volume confirmation to avoid false breakouts.
- Camarilla levels provide structured support/resistance that works in both ranging and trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels (R3, S3) on 4h using previous day's OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using 4h bar's OHLC to calculate levels for next bar
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_4h[0] - low_4h[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for volume, 14 for ATR, 50 for EMA)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.5 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.5 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > camarilla_r3_aligned[i]
        bearish_breakout = curr_close < camarilla_s3_aligned[i]
        
        # Trend filter from 12h EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 12h EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.30
                    position = 1
                    highest_high = curr_high
                    lowest_low = float('inf')
                # Short: bearish breakout AND price below 12h EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.30
                    position = -1
                    highest_high = 0.0
                    lowest_low = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 18:51
