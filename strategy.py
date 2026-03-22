#!/usr/bin/env python3
"""
Experiment #010: 1h Fisher Transform + 4h/12h HMA Trend + Volume/Session Filter

Hypothesis: 1h entries with dual-HTF trend filter (4h + 12h HMA) and Fisher Transform
reversal signals will catch trend continuations with better timing than RSI alone.
Volume and session filters reduce false signals during low-liquidity periods.

Key design:
1. 12h HMA(21) for major trend bias (call ONCE via mtf_data)
2. 4h HMA(21) for intermediate trend confirmation (call ONCE via mtf_data)
3. Fisher Transform(9) for entry timing - sharper turning points than RSI
4. Volume > 0.8x 20-bar average to confirm moves
5. Session filter: 8-20 UTC only (high liquidity hours)
6. ATR(14) stoploss at 2.5x
7. Discrete sizing: 0.25 base, 0.30 when both HTF align

Why this should work:
- 1h TF allows precise entry timing within HTF trend
- Dual HTF (4h + 12h) reduces whipsaws - both must agree for strong signal
- Fisher Transform catches reversals better than RSI in bear/range markets
- Volume + session filters cut 60% of low-quality signals
- Expected 40-60 trades/year (within 30-60 target for 1h)

Timeframe: 1h (REQUIRED)
HTF: 4h + 12h via mtf_data helper (call ONCE before loop!)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_dual_htf_hma_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for sharper turning points.
    Entry: Fisher crosses above -1.5 (long), below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range == 0:
            continue
        
        # Normalize price to 0-1 range
        x = (close[i] - lowest_low) / price_range
        
        # Clamp to avoid division issues
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    hour = pd.to_datetime(ts_seconds, unit='s').dt.hour.values
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMA trends
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # === HTF TREND BIAS (Dual: 4h + 12h) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend: both HTF agree
        strong_bullish = htf_4h_bullish and htf_12h_bullish
        strong_bearish = htf_4h_bearish and htf_12h_bearish
        
        # Weak trend: only one HTF agrees
        weak_bullish = (htf_4h_bullish or htf_12h_bullish) and not strong_bearish
        weak_bearish = (htf_4h_bearish or htf_12h_bearish) and not strong_bullish
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_ok = vol_ratio > 0.8
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = utc_hour[i]
        session_ok = 8 <= hour <= 20
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Also check for continuation: Fisher moving in trend direction
        fisher_bullish_cont = fisher[i] > fisher_trigger[i] and fisher[i] > -0.5
        fisher_bearish_cont = fisher[i] < fisher_trigger[i] and fisher[i] < 0.5
        
        # === ENTRY LOGIC - CONFLUENCE REQUIRED ===
        new_signal = 0.0
        
        # Long entries (need 3+ confluence: HTF + Fisher + Volume/Session)
        if strong_bullish and volume_ok:
            if fisher_long or (fisher_bullish_cont and session_ok):
                new_signal = STRONG_SIZE
        elif weak_bullish and volume_ok and session_ok:
            if fisher_long:
                new_signal = BASE_SIZE
        
        # Short entries
        if strong_bearish and volume_ok:
            if fisher_short or (fisher_bearish_cont and session_ok):
                new_signal = -STRONG_SIZE
        elif weak_bearish and volume_ok and session_ok:
            if fisher_short:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~40 hours on 1h), force entry if HTF aligns
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if strong_bullish and session_ok:
                new_signal = BASE_SIZE * 0.8
            elif strong_bearish and session_ok:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if both HTF turn bearish
            if position_side > 0 and strong_bearish:
                trend_reversal = True
            # Exit short if both HTF turn bullish
            if position_side < 0 and strong_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or fisher_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals