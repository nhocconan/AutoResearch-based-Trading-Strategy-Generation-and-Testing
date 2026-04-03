# Strategy: exp_221_4h_donchian_daily_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.355 | +35.2% | -9.9% | 124 | PASS |
| ETHUSDT | 0.214 | +30.6% | -14.4% | 118 | PASS |
| SOLUSDT | 0.583 | +72.1% | -21.0% | 111 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.830 | -0.5% | -4.3% | 47 | FAIL |
| SOLUSDT | 0.008 | +5.6% | -9.2% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #221: 4h Donchian(20) Breakout + Daily Trend Filter + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by daily trend (price > daily EMA50 = bullish bias, price < daily EMA50 = bearish bias) and volume spikes (>2.0x average) capture strong momentum moves with reduced false breakouts. Daily EMA50 provides dynamic structural support/resistance that adapts to market conditions. 4h timeframe targets 19-50 trades/year (75-200 total over 4 years) to minimize fee drag while capturing significant moves. ATR-based stoploss manages risk. Works in both bull (breakouts with volume) and bear (failed breaks reverse sharply).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_221_4h_donchian_daily_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 from 1d close prices
    daily_ema50 = np.full(n, np.nan)
    
    if len(df_1d) >= 50:  # Need enough data for EMA50
        # Align 1d data to LTF index
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate EMA50 on daily close
        ema50_series = pd.Series(df_1d_indexed['close'].values).ewm(span=50, min_periods=50, adjust=False).mean()
        daily_ema50_series = pd.Series(index=df_1d_indexed.index, data=ema50_series.values)
        
        # Align to LTF (4h) timeframe with shift(1) for completed bars only
        daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50_series.values)
    else:
        daily_ema50_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF daily EMA, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Daily EMA Trend Filter: Price > EMA50 = bullish bias, Price < EMA50 = bearish bias ---
        price_above_daily_ema = close[i] > daily_ema50_aligned[i]
        price_below_daily_ema = close[i] < daily_ema50_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above daily EMA50
        long_condition = breakout_up and volume_spike and price_above_daily_ema
        
        # Short: Donchian breakout down + volume spike + price below daily EMA50
        short_condition = breakout_down and volume_spike and price_below_daily_ema
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 11:36
