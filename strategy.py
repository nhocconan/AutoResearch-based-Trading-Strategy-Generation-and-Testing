#!/usr/bin/env python3
"""
Experiment #187: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Single-regime strategies fail because crypto alternates between trending
and ranging. This strategy detects regime via Choppiness Index and adapts:
- RANGE (CHOP > 55): Connors RSI mean reversion at extremes
- TREND (CHOP < 40): HMA trend + Donchian breakout + RSI pullback
- 1w HMA slope provides major trend bias (avoid counter-trend trades)

Why this should work:
- 1d timeframe = 10-30 trades/year (minimal fee drag)
- Dual regime adapts to market conditions
- 1w HTF prevents fighting major trends
- Discrete position sizing (0.25-0.30) minimizes churn
- ATR trailing stop protects capital

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 10-30/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dualregime_connors_hma_1w_v1"
timeframe = "1d"
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
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_48 = calculate_hma(close, 48)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W TREND BIAS (major trend filter) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 42
        is_neutral = not is_range_market and not is_trend_market
        
        # === 1D TREND ===
        hma_21_above_48 = hma_1d_21[i] > hma_1d_48[i]
        hma_21_below_48 = hma_1d_21[i] < hma_1d_48[i]
        hma_1d_bullish = hma_1d_slope[i] > 0.5
        hma_1d_bearish = hma_1d_slope[i] < -0.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_extreme_low = crsi[i] < 12
        crsi_extreme_high = crsi[i] > 88
        crsi_moderate_low = crsi[i] < 30
        crsi_moderate_high = crsi[i] > 70
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_neutral:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_score = 0
        long_confidence = 0
        
        # Path 1: RANGE + CRSI extreme (mean reversion)
        if is_range_market and crsi_extreme_low:
            long_score += 3
            long_confidence += 2
        
        # Path 2: RANGE + CRSI oversold + BB lower
        if is_range_market and crsi_oversold and price_below_bb_lower:
            long_score += 4
            long_confidence += 2
        
        # Path 3: TREND + weekly bullish + HMA bullish + CRSI pullback
        if is_trend_market and (weekly_bullish or price_above_1w_hma) and hma_1d_bullish and crsi_moderate_low:
            long_score += 3
            long_confidence += 2
        
        # Path 4: TREND + Donchian breakout + weekly bullish
        if is_trend_market and donchian_breakout_up and (weekly_bullish or hma_21_above_48):
            long_score += 3
            long_confidence += 1
        
        # Path 5: HMA crossover + CRSI confirmation
        if hma_21_above_48 and crsi[i] < 40 and bars_since_last_trade > 30:
            long_score += 2
            long_confidence += 1
        
        # Path 6: Simple oversold fallback (ensures trades)
        if crsi[i] < 15 and price_below_bb_lower:
            long_score += 2
            long_confidence += 1
        
        # Path 7: Weekly bullish + deep pullback
        if weekly_bullish and crsi[i] < 25 and hma_1d_bearish:
            long_score += 2
            long_confidence += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 45:
            new_signal = current_size * 0.7
        elif long_score >= 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: RANGE + CRSI extreme
        if is_range_market and crsi_extreme_high:
            short_score += 3
            short_confidence += 2
        
        # Path 2: RANGE + CRSI overbought + BB upper
        if is_range_market and crsi_overbought and price_above_bb_upper:
            short_score += 4
            short_confidence += 2
        
        # Path 3: TREND + weekly bearish + HMA bearish + CRSI pullback
        if is_trend_market and (weekly_bearish or price_below_1w_hma) and hma_1d_bearish and crsi_moderate_high:
            short_score += 3
            short_confidence += 2
        
        # Path 4: TREND + Donchian breakdown + weekly bearish
        if is_trend_market and donchian_breakout_down and (weekly_bearish or hma_21_below_48):
            short_score += 3
            short_confidence += 1
        
        # Path 5: HMA crossover + CRSI confirmation
        if hma_21_below_48 and crsi[i] > 60 and bars_since_last_trade > 30:
            short_score += 2
            short_confidence += 1
        
        # Path 6: Simple overbought fallback
        if crsi[i] > 85 and price_above_bb_upper:
            short_score += 2
            short_confidence += 1
        
        # Path 7: Weekly bearish + sharp rally
        if weekly_bearish and crsi[i] > 75 and hma_1d_bullish:
            short_score += 2
            short_confidence += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 45:
            new_signal = -current_size * 0.7
        elif short_score >= 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~120 days on 1d)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 30:
                new_signal = current_size * 0.4
            elif weekly_bearish and crsi[i] > 70:
                new_signal = -current_size * 0.4
            elif crsi[i] < 18:
                new_signal = current_size * 0.35
            elif crsi[i] > 82:
                new_signal = -current_size * 0.35
        
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
            if position_side > 0 and is_trend_market and hma_1d_bearish and weekly_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and hma_1d_bullish and weekly_bullish:
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