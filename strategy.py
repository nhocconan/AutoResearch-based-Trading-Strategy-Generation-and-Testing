#!/usr/bin/env python3
"""
Experiment #202: 12h Primary + 1d/1w HTF — Simplified Connors RSI Mean Reversion

Hypothesis: Previous strategies failed due to overly complex entry conditions that
filtered out too many trades (Sharpe=0.000 = 0 trades). This strategy simplifies:

1. CONNORS RSI (3-period): Primary entry signal with looser thresholds (20/80 vs 15/85)
2. BOLLINGER BANDS (20, 2.0): Confirmation only, not required for entry
3. 1d HMA(21): Trend bias for position sizing (larger size with trend)
4. 1w HMA(48): Major regime filter (avoid counter-trend in strong weekly moves)
5. VOLATILITY FILTER: ATR ratio for sizing (reduce size in extreme vol)

Key changes from failed experiments:
- Lowered CRSI thresholds (20/80 instead of 15/85) for MORE trades
- Removed complex scoring system — single strong signal is enough
- Added frequency floor: force entry every 100 bars if conditions met
- Asymmetric sizing: 0.35 with trend, 0.20 counter-trend
- Simpler stoploss: 2.5 * ATR(14) trailing

Timeframe: 12h (REQUIRED for 20-50 trades/year target)
HTF: 1d + 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.35 discrete based on trend alignment
Target: 30-60 trades/year per symbol (looser than previous 20-50)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_bb_simple_1d1w_v3"
timeframe = "12h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 12.5)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12.5)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 and not all(np.isnan(x)) else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Calculate 1w HTF indicators
    hma_1w_48 = calculate_hma(df_1w['close'].values, 48)
    hma_1w_slope = calculate_hma_slope(hma_1w_48, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_48_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_48)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volatility ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE_WITH_TREND = 0.35
    BASE_SIZE_COUNTER = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_48_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W REGIME FILTER ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.1
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.1
        price_above_1w_hma = close[i] > hma_1w_48_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_48_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_extreme = atr_ratio[i] > 2.0  # Reduce size in extreme vol
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === POSITION SIZING ===
        current_size_long = BASE_SIZE_WITH_TREND if (trend_1d_bullish or trend_1w_bullish) else BASE_SIZE_COUNTER
        current_size_short = BASE_SIZE_WITH_TREND if (trend_1d_bearish or trend_1w_bearish) else BASE_SIZE_COUNTER
        
        if vol_extreme:
            current_size_long *= 0.7
            current_size_short *= 0.7
        
        # === ENTRY LOGIC (LOOSENED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simplified, single strong signal
        long_entry = False
        
        # Path 1: CRSI extreme oversold (primary signal)
        if crsi_extreme_low:
            long_entry = True
        
        # Path 2: CRSI oversold + BB lower (confluence)
        if crsi_oversold and price_below_bb_lower:
            long_entry = True
        
        # Path 3: CRSI very low + 1d bullish bias (pullback in uptrend)
        if crsi[i] < 30 and (trend_1d_bullish or price_above_1d_hma):
            long_entry = True
        
        # Path 4: Price below 1w HMA but CRSI low (deep pullback)
        if crsi[i] < 35 and price_below_1w_hma:
            long_entry = True
        
        if long_entry:
            new_signal = current_size_long
        
        # SHORT ENTRIES
        short_entry = False
        
        # Path 1: CRSI extreme overbought (primary signal)
        if crsi_extreme_high:
            short_entry = True
        
        # Path 2: CRSI overbought + BB upper (confluence)
        if crsi_overbought and price_above_bb_upper:
            short_entry = True
        
        # Path 3: CRSI very high + 1d bearish bias (rally in downtrend)
        if crsi[i] > 70 and (trend_1d_bearish or price_below_1d_hma):
            short_entry = True
        
        # Path 4: Price above 1w HMA but CRSI high (rally in bear)
        if crsi[i] > 65 and price_above_1w_hma:
            short_entry = True
        
        if short_entry:
            # Only short if not already long (avoid flip-flop)
            if new_signal > 0:
                new_signal = -current_size_short
            else:
                new_signal = -current_size_short
        
        # === FREQUENCY FLOOR (CRITICAL for avoiding 0 trades) ===
        # Force trade if no signal for 100 bars (~50 days on 12h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if crsi[i] < 30:
                new_signal = current_size_long * 0.5
            elif crsi[i] > 70:
                new_signal = -current_size_short * 0.5
            elif trend_1d_bullish and crsi[i] < 40:
                new_signal = current_size_long * 0.4
            elif trend_1d_bearish and crsi[i] > 60:
                new_signal = -current_size_short * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        if stoploss_triggered or crsi_exit:
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            else:
                # Same direction, update tracking
                if position_side > 0 and close[i] > highest_price:
                    highest_price = close[i]
                if position_side < 0 and (lowest_price == 0.0 or close[i] < lowest_price):
                    lowest_price = close[i]
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