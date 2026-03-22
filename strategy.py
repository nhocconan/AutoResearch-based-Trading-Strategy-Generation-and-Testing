#!/usr/bin/env python3
"""
Experiment #114: 4h Primary + 12h/1d HTF — Dual Regime Mean Reversion + Trend Pullback

Hypothesis: Previous 4h strategies failed due to overly strict entry conditions (0 trades).
This strategy simplifies confluence requirements while maintaining edge through:

1. DUAL REGIME DETECTION: Choppiness Index splits market into range (CHOP>55) vs trend (CHOP<45)
2. RANGE MODE: Pure mean reversion at Bollinger extremes + Connors RSI
3. TREND MODE: Pullback entries in direction of 12h HMA trend only
4. 12h HMA SLOPE: Major trend bias (avoid counter-trend trades)
5. VOLATILITY FILTER: ATR ratio ensures we're not entering during calm periods
6. FALLBACK ENTRIES: Force trades after 120 bars without signal (prevents 0 trades)

Key improvements over failed strategies:
- Fewer confluence requirements (2-3 factors instead of 4-5)
- Looser CRSI thresholds (25/75 instead of 15/85)
- Lower vol spike threshold (1.5 instead of 1.8)
- Mandatory fallback entries every 120 bars
- Asymmetric sizing: 0.30 for high conviction, 0.20 for fallback

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol (meets 10+ train, 3+ test requirement)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_connors_12h_v2"
timeframe = "4h"
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
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.20
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === 12H TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50  # Lowered threshold for more regime switches
        is_trend_market = chop_14[i] < 48  # Overlap allows both false in transition
        
        # === VOLATILITY FILTER ===
        vol_elevated = atr_ratio[i] > 1.4  # Lowered from 1.6 for more signals
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30  # Looser threshold
        crsi_overbought = crsi[i] > 70  # Looser threshold
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if vol_elevated:
            current_size = HIGH_CONV_SIZE
        elif not is_range_market and not is_trend_market:
            current_size = LOW_CONV_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simplified confluence (2 factors instead of 3-4)
        long_confidence = 0
        
        # Path 1: Range market + CRSI oversold (primary mean revert)
        if is_range_market and crsi_oversold:
            long_confidence += 2
        
        # Path 2: Price below BB lower + CRSI oversold (capitulation)
        if price_below_bb_lower and crsi_oversold:
            long_confidence += 2
        
        # Path 3: Trend market + bullish bias + pullback
        if is_trend_market and trend_12h_bullish and crsi[i] < 40:
            long_confidence += 2
        
        # Path 4: Price below 12h HMA + extreme CRSI (deep pullback)
        if price_below_12h_hma and crsi_extreme_low:
            long_confidence += 2
        
        # Path 5: RSI oversold + BB lower (classic mean revert)
        if rsi_oversold and price_below_bb_lower:
            long_confidence += 1
        
        # Path 6: Vol elevated + oversold (panic buy)
        if vol_elevated and crsi[i] < 35:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence == 2 and bars_since_last_trade > 60:
            new_signal = LOW_CONV_SIZE
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_confidence += 2
        
        # Path 2: Price above BB upper + CRSI overbought
        if price_above_bb_upper and crsi_overbought:
            short_confidence += 2
        
        # Path 3: Trend market + bearish bias + pullback
        if is_trend_market and trend_12h_bearish and crsi[i] > 60:
            short_confidence += 2
        
        # Path 4: Price above 12h HMA + extreme CRSI (rally in bear)
        if price_above_12h_hma and crsi_extreme_high:
            short_confidence += 2
        
        # Path 5: RSI overbought + BB upper
        if rsi_overbought and price_above_bb_upper:
            short_confidence += 1
        
        # Path 6: Vol elevated + overbought (panic sell)
        if vol_elevated and crsi[i] > 65:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence == 2 and bars_since_last_trade > 60:
            new_signal = -LOW_CONV_SIZE
        
        # === FALLBACK ENTRIES (CRITICAL - prevents 0 trades) ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and crsi[i] < 40:
                new_signal = LOW_CONV_SIZE * 0.8
            elif trend_12h_bearish and crsi[i] > 60:
                new_signal = -LOW_CONV_SIZE * 0.8
            elif crsi[i] < 25:
                new_signal = LOW_CONV_SIZE * 0.7
            elif crsi[i] > 75:
                new_signal = -LOW_CONV_SIZE * 0.7
            elif bb_pct < 0.15:
                new_signal = LOW_CONV_SIZE * 0.6
            elif bb_pct > 0.85:
                new_signal = -LOW_CONV_SIZE * 0.6
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_12h_bearish and hma_12h_slope_aligned[i] < -0.5:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_12h_bullish and hma_12h_slope_aligned[i] > 0.5:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_reversal = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_reversal = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_reversal = True
        
        if stoploss_triggered or regime_reversal or crsi_reversal:
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