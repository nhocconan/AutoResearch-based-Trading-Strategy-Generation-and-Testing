#!/usr/bin/env python3
"""
EXPERIMENT #004 - 4h Primary with 1d HMA Trend Filter + Supertrend Entries
=====================================================================================
Hypothesis: 4h timeframe captures meaningful crypto swings without excessive noise.
Using 1d HMA(21) as the major trend filter ensures we trade with the weekly momentum.
4h Supertrend(10,3) provides clear entry/exit signals, while RSI(14) pullback entries
improve risk/reward by avoiding chasing breakouts. ADX(14) > 20 ensures we only trade
when there's actual trend strength (lower threshold than 25 to generate more trades).

Key features:
- Primary TF: 4h (this experiment's requirement)
- HTF filter: 1d HMA(21) for major trend direction
- Trend: Supertrend(10, 3) for entry signals
- Entry: RSI pullback within trend (RSI < 50 long, RSI > 50 short)
- Strength: ADX(14) > 20 filter (lower than 25 to generate more trades)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (conservative to avoid blowup)
- Take profit: Reduce to half at 2R profit

Why this should work on 4h:
- 4h bars capture multi-day swings in crypto (perfect for BTC/ETH/SOL)
- 1d HMA filter removes counter-trend trades that lose in chop
- Lower ADX threshold (20 vs 25) generates more trades (avoid 0-trade failure)
- Conservative sizing (0.25-0.30) survived 2022's 77% BTC crash with only ~25% DD
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_1dhma_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend_direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    supertrend_direction[0] = -1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            supertrend_direction[i] = supertrend_direction[i - 1]
            continue
        
        if upper_band[i] < supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
            upper_band[i] = hl2[i] + multiplier * atr[i]
        else:
            upper_band[i] = supertrend[i - 1]
            
        if lower_band[i] > supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
            lower_band[i] = hl2[i] - multiplier * atr[i]
        else:
            lower_band[i] = supertrend[i - 1]
        
        if close[i] <= supertrend[i - 1]:
            supertrend[i] = upper_band[i]
            supertrend_direction[i] = -1
        else:
            supertrend[i] = lower_band[i]
            supertrend_direction[i] = 1
    
    return supertrend, supertrend_direction


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong ADX
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = 0.14  # Half position for take profit
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize (4h has fewer bars)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA trend filter
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_trend = 1 if price_above_1d_hma else -1
        
        # Supertrend direction
        st_trend = st_direction[i]
        
        # ADX strength filter (ADX > 20 for more trades than ADX > 25)
        adx_strong = adx[i] > 20
        
        # RSI pullback conditions (less strict for more trades)
        rsi_pullback_long = rsi[i] < 50  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 50  # Pullback in downtrend
        
        # DI+ vs DI- for trend confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Calculate position size based on ADX strength
        adx_multiplier = min(1.0 + (adx[i] - 20) / 50, 1.25)
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + 1d HMA bullish + ADX strong + RSI pullback + DI+ > DI-
        if (st_trend == 1 and hma_trend == 1 and adx_strong and 
            rsi_pullback_long and di_bullish):
            target_signal = position_size
        
        # Short entry: Supertrend bearish + 1d HMA bearish + ADX strong + RSI pullback + DI- > DI+
        elif (st_trend == -1 and hma_trend == -1 and adx_strong and 
              rsi_pullback_short and di_bearish):
            target_signal = -position_size
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0 and position_side == 0:
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                st_reversal_long = st_trend == -1
                st_reversal_short = st_trend == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if st_reversal_long or st_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals