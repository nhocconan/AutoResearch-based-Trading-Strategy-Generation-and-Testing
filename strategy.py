#!/usr/bin/env python3
"""
Experiment #023: 4h EMA Crossover + Daily Trend Filter + Volume + CHOP

HYPOTHESIS: Multi-timeframe EMA crossover with HTF trend alignment will reduce
whipsaw trades compared to simple price channel breakouts.

WHY IT SHOULD WORK IN BOTH MARKETS:
1. 4h EMA(21/63) crossover captures medium-term trend shifts
2. 21-bar EMA on 4h = ~3.5 day smooth trend, filters noise
3. Daily trend filter (from 4h data) aligns entries with HTF direction
4. CHOP < 50 avoids ranging periods where EMAs oscillate
5. Volume spike confirms institutional participation

ENTRY CONDITIONS (4-way confluence):
- Daily trend aligned (close > 21-bar EMA for longs, < for shorts)
- 4h EMA fast(21) crosses above slow(63) for longs, opposite for shorts
- Volume > 1.5x 20-bar MA (institutional confirmation)
- CHOP < 50 (not choppy, trending environment)

EXIT: 2.5 ATR stoploss, opposite EMA crossover, min 2 bars

TRADE COUNT ESTIMATE:
- 4h bars/4yr ≈ 8760 per symbol
- EMA(21/63) crossover: ~1 per 2-3 weeks = ~26-39 raw signals/year
- Daily trend filter: ~50% pass = ~13-20/year
- Volume + CHOP filter: ~50% pass = ~7-10/year per symbol
- TARGET: 75-120 total over 4 years (19-30/year)
- In acceptable range, conservative but high-quality signals
"""
import numpy as np
import pandas as pd

name = "mtf_4h_ema_cross_daily_trend_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized after TR calculation"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - regime filter
    CHOP > 61.8 = choppy/range (avoid entries)
    CHOP < 50 = trending (allow entries)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        period_high = high[i-period+1:i+1].max()
        period_low = low[i-period+1:i+1].min()
        
        if period_high > period_low:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            if period_high != period_low:
                chop[i] = 100 * np.log10(sum_tr / (period_high - period_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # EMA(21) fast and EMA(63) slow for crossover signal
    ema_fast = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=63, min_periods=63, adjust=False).mean().values
    
    # Daily trend filter: 21-bar EMA from 4h data (~3.5 day trend)
    ema_daily_trend = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 65  # max(63 for EMA63, 20 for vol, 14 for ATR)
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: CHOP < 50 (trending, not choppy) ===
        chop_ok = chop[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] > 1.5
        
        # === EMA CROSSOVER on 4h ===
        bullish_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        bearish_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === DAILY TREND FILTER (from 4h EMA - ~3.5 day trend) ===
        daily_bull = close[i] > ema_daily_trend[i]
        daily_bear = close[i] < ema_daily_trend[i]
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Trend reversal - opposite EMA crossover
            reversal_exit = (position_side > 0 and min_hold and bearish_cross) or \
                           (position_side < 0 and min_hold and bullish_cross)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif reversal_exit:
                # Exit and potentially reverse
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if choppy market
            if not chop_ok:
                signals[i] = 0.0
                continue
            
            # LONG: EMA bullish cross + daily trend aligned + volume spike
            if bullish_cross and daily_bull and vol_ok:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: EMA bearish cross + daily trend aligned + volume spike
            elif bearish_cross and daily_bear and vol_ok:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals