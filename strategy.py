#!/usr/bin/env python3
"""
Experiment #035: 1h Fisher Transform Entries + 4h/1d HMA Trend Filter + Session

Hypothesis: Ehlers Fisher Transform provides cleaner reversal signals than RSI in bear/range markets.
Key design based on learned failures from 27+ experiments:
1. 1h primary timeframe (REQUIRED for this experiment)
2. 4h HMA(21) for intermediate trend direction (call ONCE before loop via mtf_data)
3. 1d HMA(21) for major trend bias (call ONCE before loop via mtf_data)
4. Fisher Transform(9) for entry timing - crosses at -1.0/+1.0 are cleaner than RSI
5. Session filter 8-20 UTC only (high liquidity, reduces false signals)
6. Volume > 0.5x average (loose filter to ensure trades generate)
7. ATR(14) stoploss at 2.5x - protects from major drawdowns
8. Discrete sizing: 0.20 base, 0.25 medium, 0.30 strong alignment

Why this should work (different from failed 1h strategies #025, #028, #030):
- Fisher Transform catches reversals better than RSI in bear markets (research-backed)
- Loose volume filter (0.5x not 0.8x) ensures trades actually trigger
- Session filter reduces noise but doesn't kill all signals
- 4h+1d dual HTF filter provides strong trend confirmation without being too strict
- Fisher thresholds (-1.0/+1.0) are looser than typical (-1.5/+1.5) to ensure trade generation

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
Target trades: 30-80/year (≈0.25/day on 1h data)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_1d_hma_session_v2"
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
    Converts price to Gaussian distribution for clearer reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = EMA(price, period) normalized
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize typical price over lookback period
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1 range
        x = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag of Fisher)
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMA trends
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    volume_sma_20 = calculate_volume_sma(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    MEDIUM_SIZE = 0.25
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(volume_sma_20[i]) or volume_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER (loose - 0.5x average) ===
        volume_ok = volume[i] > 0.5 * volume_sma_20[i]
        
        # === HTF TREND BIAS (4h + 1d) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 4h and 1d agree
        htf_strong_bullish = htf_4h_bullish and htf_1d_bullish
        htf_strong_bearish = htf_4h_bearish and htf_1d_bearish
        
        # Weak trend: at least one HTF agrees
        htf_weak_bullish = htf_4h_bullish or htf_1d_bullish
        htf_weak_bearish = htf_4h_bearish or htf_1d_bearish
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.0 (oversold reversal)
        fisher_long = fisher_signal[i] < -1.0 and fisher[i] > -1.0
        
        # Short: Fisher crosses below +1.0 (overbought reversal)
        fisher_short = fisher_signal[i] > 1.0 and fisher[i] < 1.0
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_strong_bullish:
            current_size = STRONG_SIZE
        elif htf_weak_bullish:
            current_size = MEDIUM_SIZE
        elif htf_strong_bearish:
            current_size = STRONG_SIZE
        elif htf_weak_bearish:
            current_size = MEDIUM_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (confluence required but not too strict) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + Fisher reversal + session + volume
        if htf_weak_bullish and fisher_long and in_session and volume_ok:
            new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + Fisher reversal + session + volume
        elif htf_weak_bearish and fisher_short and in_session and volume_ok:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 48 bars (~2 days on 1h), allow weaker entry (no session filter)
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if htf_weak_bullish and fisher_long and volume_ok:
                new_signal = current_size * 0.8
            elif htf_weak_bearish and fisher_short and volume_ok:
                new_signal = -current_size * 0.8
        
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
            # Exit long if 4h trend turns bearish
            if position_side > 0 and htf_4h_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and htf_4h_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -1.5:
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