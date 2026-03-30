#!/usr/bin/env python3
"""
Experiment #022: 1d Williams %R Cross + Weekly HTF + SMA Filter (1d primary)

HYPOTHESIS: Williams %R crossing OUT of extreme territory with weekly trend
confirmation catches mean reversion trades aligned with larger timeframe.
%R crossing = more selective than just extreme reading = fewer but better trades.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: %R crosses above -80 (recovering from oversold) + price>SMA50 + weekly bull
- Bear: %R crosses below -20 (falling from overbought) + price<SMA50 + weekly bear
- HTF trend alignment prevents catching knives
- Volume confirms institutional interest

Keep rate target: 45-55%. Total trades: 50-80 over 4 years (12-20/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_willr_cross_weekly_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF weekly ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA for trend direction
    ema_fast_w = pd.Series(df_1w['close'].values).ewm(span=8, min_periods=8, adjust=False).mean()
    ema_slow_w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean()
    htf_bull = ema_fast_w > ema_slow_w
    htf_bear = ema_fast_w < ema_slow_w
    
    # Align HTF to LTF with proper shift(1) to avoid look-ahead
    htf_bull_aligned = align_htf_to_ltf(prices, df_1w, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1w, htf_bear.astype(float))
    
    # === Local 1d indicators ===
    
    # Williams %R (14) - more selective with cross detection
    period_wr = 14
    willr = np.full(n, np.nan)
    for i in range(period_wr, n):
        hh = np.max(high[i-period_wr+1:i+1])
        ll = np.min(low[i-period_wr+1:i+1])
        if hh != ll:
            willr[i] = -100 * (hh - close[i]) / (hh - ll)
    
    # ATR (14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # SMA 50 for directional filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume ratio (1.5x minimum)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signal generation ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_pos = False
    pos_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    cooldown = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Validate indicators
        if np.isnan(willr[i]) or np.isnan(sma50[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(htf_bull_aligned[i]) or np.isnan(htf_bear_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === Williams %R Cross detection ===
        # %R crossing ABOVE -80 = leaving oversold = potential long
        # %R crossing BELOW -20 = entering overbought = potential short
        willr_long_cross = willr[i] > -80 and willr[i-1] <= -80
        willr_short_cross = willr[i] < -20 and willr[i-1] >= -20
        
        # === Price vs SMA50 ===
        price_above_sma = close[i] > sma50[i]
        price_below_sma = close[i] < sma50[i]
        
        # === HTF alignment (requires trend confirmation) ===
        htf_bull_ok = htf_bull_aligned[i] > 0.5
        htf_bear_ok = htf_bear_aligned[i] > 0.5
        
        # === Volume confirmation ===
        vol_ok = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        # === Entry logic ===
        if not in_pos and cooldown <= 0:
            # LONG: %R crosses above -80 + price above SMA50 + weekly bull + vol
            if willr_long_cross and price_above_sma and htf_bull_ok and vol_ok:
                desired_signal = SIZE
            
            # SHORT: %R crosses below -20 + price below SMA50 + weekly bear + vol
            elif willr_short_cross and price_below_sma and htf_bear_ok and vol_ok:
                desired_signal = -SIZE
        
        # === Exit logic ===
        if in_pos:
            if pos_side > 0:
                # Long: 2.5 ATR stop
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = SIZE
                    
            elif pos_side < 0:
                # Short: 2.5 ATR stop
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = -SIZE
        
        # === Minimum hold: 3 bars to reduce fee churn ===
        if in_pos and (i - entry_bar) < 3:
            desired_signal = pos_side * SIZE
        
        # === Update position ===
        if desired_signal != 0.0:
            if not in_pos or np.sign(desired_signal) != pos_side:
                in_pos = True
                pos_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
        else:
            if in_pos:
                in_pos = False
                pos_side = 0
                cooldown = 3  # 3-bar cooldown after exit
        
        if cooldown > 0:
            cooldown -= 1
        
        signals[i] = desired_signal
    
    return signals