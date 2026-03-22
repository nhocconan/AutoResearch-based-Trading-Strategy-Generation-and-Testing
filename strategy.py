#!/usr/bin/env python3
"""
Experiment #164: 4h Primary + 12h/1d HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 4h strategies failed due to overly complex entry conditions
that filtered out too many trades. This strategy uses a cleaner dual-regime approach:

1. CHOPPINESS INDEX regime detection (CHOP > 55 = range, CHOP < 45 = trend)
2. RANGE REGIME: Connors RSI mean reversion at extremes (CRSI < 20 long, > 80 short)
3. TREND REGIME: Pullback entries aligned with 12h HMA trend direction
4. 1d HMA slope for major bias filter (avoid counter-trend in strong moves)
5. Simplified confluence: max 2-3 conditions to ensure trade generation

Why this should work:
- 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
- Dual regime adapts to market conditions (range vs trend)
- 12h/1d HTF prevents fighting major trends
- Simpler entry logic = more trades generated
- Asymmetric sizing: larger positions with HTF trend

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_connors_hma_12h1d_v1"
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.32  # Larger size when aligned with HTF trend
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 18
        crsi_extreme_high = crsi[i] > 82
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        bars_since_last_trade = i - last_trade_bar
        
        # === RANGE REGIME: MEAN REVERSION ===
        long_range = False
        short_range = False
        
        if is_range_market:
            # Long: CRSI extreme low + BB lower or RSI oversold
            if crsi_extreme_low and (price_below_bb_lower or rsi_oversold):
                long_range = True
            elif crsi_oversold and price_below_bb_lower:
                long_range = True
            
            # Short: CRSI extreme high + BB upper or RSI overbought
            if crsi_extreme_high and (price_above_bb_upper or rsi_overbought):
                short_range = True
            elif crsi_overbought and price_above_bb_upper:
                short_range = True
        
        # === TREND REGIME: PULLBACK ENTRIES ===
        long_trend = False
        short_trend = False
        
        if is_trend_market:
            # Long: 12h bullish + pullback (CRSI low but not extreme)
            if trend_12h_bullish and crsi[i] < 40 and price_above_12h_hma:
                long_trend = True
            elif trend_12h_bullish and crsi_oversold:
                long_trend = True
            
            # Short: 12h bearish + pullback (CRSI high but not extreme)
            if trend_12h_bearish and crsi[i] > 60 and price_below_12h_hma:
                short_trend = True
            elif trend_12h_bearish and crsi_overbought:
                short_trend = True
        
        # === NEUTRAL REGIME: SIMPLER ENTRIES ===
        long_neutral = False
        short_neutral = False
        
        if not is_range_market and not is_trend_market:
            # Simple CRSI extremes with HTF bias
            if crsi_extreme_low and not trend_1d_bearish:
                long_neutral = True
            elif crsi_oversold and trend_12h_bullish:
                long_neutral = True
            
            if crsi_extreme_high and not trend_1d_bullish:
                short_neutral = True
            elif crsi_overbought and trend_12h_bearish:
                short_neutral = True
        
        # === DETERMINE SIGNAL ===
        new_signal = 0.0
        
        # Long signal priority
        if long_range:
            new_signal = BASE_SIZE
        elif long_trend:
            new_signal = TREND_SIZE if trend_12h_bullish else BASE_SIZE
        elif long_neutral:
            new_signal = BASE_SIZE * 0.8
        
        # Short signal priority (overrides long if stronger)
        if short_range:
            new_signal = -BASE_SIZE
        elif short_trend:
            new_signal = -TREND_SIZE if trend_12h_bearish else -BASE_SIZE
        elif short_neutral:
            new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD: Force trades if none for long period ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            # Simpler conditions to ensure trades
            if crsi[i] < 22 and not trend_1d_bearish:
                new_signal = BASE_SIZE * 0.5
            elif crsi[i] > 78 and not trend_1d_bullish:
                new_signal = -BASE_SIZE * 0.5
            elif trend_12h_bullish and crsi[i] < 35:
                new_signal = BASE_SIZE * 0.4
            elif trend_12h_bearish and crsi[i] > 65:
                new_signal = -BASE_SIZE * 0.4
        
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
            # Exit long if regime shifts to strong trend bearish
            if position_side > 0 and is_trend_market and trend_12h_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and trend_12h_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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