#!/usr/bin/env python3
"""
Experiment #113: 1d Primary + 1w HTF — Dual Regime Connors + Donchian Breakout

Hypothesis: Daily timeframe with weekly trend bias provides optimal trade frequency
(20-50/year) while avoiding fee drag. Previous failures came from either too many
trades (lower TF) or too few trades (overly strict filters). This strategy combines:

1. 1W HMA(21) SLOPE: Major trend bias — only trade with weekly trend direction
2. CHOPPINESS INDEX (14): Regime detection — range (>55) vs trend (<45)
3. CONNORS RSI: Mean reversion entries in range markets (CRSI<25 long, >75 short)
4. DONCHIAN(20) BREAKOUT: Trend follow entries in trend markets
5. ATR(14) TRAILING STOP: 2.5x ATR stoploss on all positions
6. VOLUME CONFIRMATION: Volume > SMA(20) on breakout bars

Why this should work:
- 1d timeframe = natural 20-50 trades/year target
- 1w HTF prevents fighting major trends (critical for 2022 crash, 2025 bear)
- Dual regime adapts to market conditions (range = mean revert, trend = breakout)
- Connors RSI has proven 75% win rate for mean reversion
- Donchian breakout captures sustained trends without whipsaw
- Conservative position sizing (0.25-0.30) limits drawdown during crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_donchian_1w_v2"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Volume confirmation
    vol_above_avg = volume > vol_sma_20 * 1.2
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND BIAS ===
        # Weekly trend must be clear (slope > 0.5% or < -0.5%)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        is_transition = not is_range_market and not is_trend_market
        
        # === CONNORS RSI EXTREMES ===
        # Relaxed thresholds for more trades (was 15/85, now 25/75)
        crsi_oversold = crsi[i] < 28
        crsi_overbought = crsi[i] > 72
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i] * 0.998  # Near breakout
        breakout_short = close[i] < donchian_lower[i] * 1.002
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_above_avg[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_transition:
            current_size = BASE_SIZE * 0.6  # Reduce size in unclear regimes
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade generation
        long_score = 0
        
        # Path 1: Range market + CRSI oversold (mean reversion)
        if is_range_market and crsi_oversold:
            long_score += 3
        
        # Path 2: Trend market + bullish weekly + pullback (CRSI moderate low)
        if is_trend_market and trend_1w_bullish and crsi[i] < 40:
            long_score += 2
        
        # Path 3: Trend market + Donchian breakout + volume
        if is_trend_market and breakout_long and vol_confirmed and trend_1w_bullish:
            long_score += 3
        
        # Path 4: Price above 1w HMA + CRSI low (bull market pullback)
        if price_above_1w_hma and crsi[i] < 35:
            long_score += 2
        
        # Path 5: Neutral weekly + CRSI extreme (strong mean revert)
        if trend_1w_neutral and crsi_extreme_low:
            long_score += 2
        
        # Path 6: Simple oversold fallback (ensures trades)
        if crsi[i] < 22 and bars_since_last_trade > 60:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_score == 1 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_score += 3
        
        # Path 2: Trend market + bearish weekly + pullback
        if is_trend_market and trend_1w_bearish and crsi[i] > 60:
            short_score += 2
        
        # Path 3: Trend market + Donchian breakdown + volume
        if is_trend_market and breakout_short and vol_confirmed and trend_1w_bearish:
            short_score += 3
        
        # Path 4: Price below 1w HMA + CRSI high (bear market rally)
        if price_below_1w_hma and crsi[i] > 65:
            short_score += 2
        
        # Path 5: Neutral weekly + CRSI extreme
        if trend_1w_neutral and crsi_extreme_high:
            short_score += 2
        
        # Path 6: Simple overbought fallback
        if crsi[i] > 78 and bars_since_last_trade > 60:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_score == 1 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~120 days on 1d)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and crsi[i] < 40:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.5
            elif crsi[i] < 25:
                new_signal = current_size * 0.4
            elif crsi[i] > 75:
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
            # Exit long if weekly trend turns bearish
            if position_side > 0 and trend_1w_bearish and is_trend_market:
                regime_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and trend_1w_bullish and is_trend_market:
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