#!/usr/bin/env python3
"""
Experiment #109: 4h Primary + 1d HTF — Adaptive KAMA Trend + Connors RSI Mean Reversion

Hypothesis: Previous 4h strategies failed because they used static indicators (EMA/HMA) that
don't adapt to changing market volatility. KAMA (Kaufman Adaptive Moving Average) adjusts
its smoothing based on market efficiency - smooth in trends, fast in ranges. Combined with:

1. KAMA(10) adaptive trend - follows price in trends, flattens in chop
2. Choppiness Index regime - CHOP>55 mean revert, CHOP<45 trend follow
3. Connors RSI(3,2,100) - entry timing with 75% win rate literature
4. 1d HMA(21) slope - major trend bias from HTF
5. Donchian(20) breakout - confirmation for trend entries
6. ATR(14) trailing stop - 2.5x for risk management

Why this should work:
- KAMA adapts to volatility regime automatically (no parameter tuning)
- Dual regime logic: mean revert in chop, trend pullback in trends
- 4h timeframe = 20-50 trades/year target (low fee drag)
- 1d HTF prevents fighting major trends
- Lenient entry thresholds ensure sufficient trade generation

Timeframe: 4h (REQUIRED for Experiment #109)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_connors_chop_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period).values)
    volatility = pd.Series(np.abs(close_s.diff().values)).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = price_change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 12)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) > 0 else 50
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
    # KAMA slope for trend direction
    kama_slope = np.zeros(n)
    for i in range(10, n):
        if kama_10[i-10] != 0:
            kama_slope[i] = (kama_10[i] - kama_10[i-10]) / kama_10[i-10] * 100
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_slope[i] > 0.1
        kama_bearish = kama_slope[i] < -0.1
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50  # More lenient for more trades
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donchian_breakdown_dn = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30  # More lenient threshold
        crsi_overbought = crsi[i] > 70
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths for more trades
        long_score = 0
        
        # Path 1: Range market + CRSI oversold (mean reversion)
        if is_range_market and crsi_oversold:
            long_score += 3
        
        # Path 2: Range market + price below BB lower
        if is_range_market and price_below_bb_lower:
            long_score += 2
        
        # Path 3: Trend market + pullback to KAMA + 1d bullish
        if is_trend_market and price_below_kama and trend_1d_bullish and crsi[i] < 40:
            long_score += 3
        
        # Path 4: 1d bullish + CRSI low (pullback in bull trend)
        if trend_1d_bullish and crsi[i] < 35:
            long_score += 2
        
        # Path 5: Price below KAMA + CRSI extreme (deep pullback)
        if price_below_kama and crsi_extreme_low:
            long_score += 2
        
        # Path 6: Donchian breakout + trend confirmation
        if donchian_breakout_up and kama_bullish and trend_1d_bullish:
            long_score += 2
        
        # Path 7: Simple oversold fallback (ensures trades)
        if crsi[i] < 25 and price_below_bb_lower:
            long_score += 2
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.67  # ~0.20
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_score += 3
        
        # Path 2: Range market + price above BB upper
        if is_range_market and price_above_bb_upper:
            short_score += 2
        
        # Path 3: Trend market + pullback to KAMA + 1d bearish
        if is_trend_market and price_above_kama and trend_1d_bearish and crsi[i] > 60:
            short_score += 3
        
        # Path 4: 1d bearish + CRSI high (rally in bear trend)
        if trend_1d_bearish and crsi[i] > 65:
            short_score += 2
        
        # Path 5: Price above KAMA + CRSI extreme
        if price_above_kama and crsi_extreme_high:
            short_score += 2
        
        # Path 6: Donchian breakdown + trend confirmation
        if donchian_breakdown_dn and kama_bearish and trend_1d_bearish:
            short_score += 2
        
        # Path 7: Simple overbought fallback
        if crsi[i] > 75 and price_above_bb_upper:
            short_score += 2
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.67
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 100 bars (~17 days on 4h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and crsi[i] < 40:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.5
            elif crsi[i] < 28:
                new_signal = current_size * 0.4
            elif crsi[i] > 72:
                new_signal = -current_size * 0.4
        
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
            # Exit long if regime shifts to strong bear trend
            if position_side > 0 and is_trend_market and trend_1d_bearish and kama_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong bull trend
            if position_side < 0 and is_trend_market and trend_1d_bullish and kama_bullish:
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