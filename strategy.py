#!/usr/bin/env python3
"""
Experiment #110: 1h Primary + 4h HTF — Regime-Adaptive Mean Reversion with Connors RSI

Hypothesis: Previous 1h strategies failed with 0 trades because entry conditions were too strict.
This strategy uses looser thresholds, multiple entry paths, and fallback logic to ensure
30-60 trades/year while maintaining edge through HTF trend filter + regime detection.

Key Components:
1. 4h HMA(21) slope for major trend bias (called ONCE before loop via mtf_data)
2. Choppiness Index(14) for regime detection (range vs trend)
3. Connors RSI for entry timing (looser thresholds: 25/75 instead of 15/85)
4. Bollinger Bands(20, 2.0) for extreme detection
5. Volume filter (loose: > 0.7x avg) to confirm moves
6. Multiple entry paths to ensure trade generation
7. Fallback entries after 100 bars without signal

Timeframe: 1h (REQUIRED for experiment #110)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.25 (smaller for lower TF to reduce fee drag)
Target trades: 40-80/year per symbol (1h sweet spot)
Stoploss: 2.2 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_connors_hma4h_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    
    # Normalize streak to 0-100 scale
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 12)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

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
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / abs(hma_values[i - lookback]) * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Price position in BB
    bb_position = (close - bb_lower) / np.where((bb_upper - bb_lower) > 0, (bb_upper - bb_lower), 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.30 for 1h)
    BASE_SIZE = 0.22
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_position[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_extreme_low = bb_position[i] < 0.1
        bb_extreme_high = bb_position[i] > 0.9
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30
        crsi_overbought = crsi[i] > 70
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === VOLUME CONFIRMATION (loose) ===
        volume_ok = vol_ratio[i] > 0.6  # Very loose to ensure trades
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE * 1.1  # Slightly larger in range (mean revert works better)
        else:
            current_size = BASE_SIZE * 0.9
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (LOOSENED for more trades)
        long_score = 0
        
        # Path 1: Range market + CRSI oversold (primary mean revert)
        if is_range_market and crsi_oversold:
            long_score += 3
        
        # Path 2: BB extreme low + CRSI oversold
        if bb_extreme_low and crsi_oversold:
            long_score += 3
        
        # Path 3: 4h bullish + pullback (trend follow entry)
        if trend_4h_bullish and crsi[i] < 40 and price_above_4h_hma:
            long_score += 2
        
        # Path 4: Price below 4h HMA but CRSI very low (deep pullback)
        if price_below_4h_hma and crsi_extreme_low:
            long_score += 2
        
        # Path 5: RSI oversold + CRSI low (double confirmation)
        if rsi_oversold and crsi[i] < 35:
            long_score += 2
        
        # Path 6: Simple CRSI extreme (fallback for trades)
        if crsi_extreme_low:
            long_score += 1
        
        # Path 7: BB lower break + volume
        if price_below_bb_lower and volume_ok:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_score += 3
        
        # Path 2: BB extreme high + CRSI overbought
        if bb_extreme_high and crsi_overbought:
            short_score += 3
        
        # Path 3: 4h bearish + pullback
        if trend_4h_bearish and crsi[i] > 60 and price_below_4h_hma:
            short_score += 2
        
        # Path 4: Price above 4h HMA but CRSI very high (rally in bear)
        if price_above_4h_hma and crsi_extreme_high:
            short_score += 2
        
        # Path 5: RSI overbought + CRSI high
        if rsi_overbought and crsi[i] > 65:
            short_score += 2
        
        # Path 6: Simple CRSI extreme
        if crsi_extreme_high:
            short_score += 1
        
        # Path 7: BB upper break + volume
        if price_above_bb_upper and volume_ok:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.6
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~5 days on 1h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 40:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.5
            elif crsi[i] < 25:
                new_signal = current_size * 0.4
            elif crsi[i] > 75:
                new_signal = -current_size * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.2 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.2 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.2 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_4h_bearish and chop_14[i] < 40:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_4h_bullish and chop_14[i] < 40:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI goes overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True
            # Exit short when CRSI goes oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True
        
        if stoploss_triggered or regime_reversal or crsi_exit:
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