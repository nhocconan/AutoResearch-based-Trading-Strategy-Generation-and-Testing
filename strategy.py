#!/usr/bin/env python3
"""
Experiment #177: 1d Primary + 1w HTF — Regime-Adaptive Dual Strategy

Hypothesis: Previous 1d strategies failed because they were too restrictive (0 trades)
or too trend-focused (negative Sharpe in bear markets). This strategy combines:

1. REGIME DETECTION: Choppiness Index + ADX to distinguish range vs trend
2. MEAN REVERSION (range regime): Connors RSI extremes + BB bands for reversals
3. TREND FOLLOWING (trend regime): Donchian breakouts + HMA confirmation
4. 1w HTF BIAS: Major trend direction to avoid counter-trend trades
5. VOLATILITY FILTER: ATR ratio to avoid entering during extreme vol spikes

Why this should work on 1d:
- 1d timeframe = 20-50 trades/year target (low fee drag, matches experiment goal)
- 1w HTF provides major trend bias without over-filtering
- Dual regime logic adapts to market conditions (bull/bear/range)
- Multiple entry paths ensure sufficient trades (avoid 0-trade failure)
- Conservative sizing (0.25-0.30) limits drawdown during 2022 crash

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol (≈80-200 over 4-year train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_dual_connors_donchian_1w_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high_s.diff().values)
    tr3 = np.abs(low_s.diff().values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * plus_di / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * minus_di / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.fillna(50).values
    
    # Component 2: RSI of Streak
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volatility ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W TREND BIAS ===
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === REGIME DETECTION ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45 and adx_14[i] > 20
        
        # === VOLATILITY FILTER ===
        vol_extreme = atr_ratio[i] > 2.0  # Avoid entering during extreme vol
        
        # === PRICE POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_below_donchian = close[i] < donchian_lower[i]
        price_above_donchian = close[i] > donchian_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        
        # === HMA TREND ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if vol_extreme:
            current_size = BASE_SIZE * 0.5  # Reduce size during extreme vol
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades
        long_score = 0
        
        # Path 1: Range market + CRSI oversold (mean reversion)
        if is_range_market and crsi_oversold:
            long_score += 3
        
        # Path 2: Range market + price below BB lower
        if is_range_market and price_below_bb_lower:
            long_score += 2
        
        # Path 3: Trend market + pullback + 1w bullish
        if is_trend_market and price_above_1w_hma and crsi[i] < 40:
            long_score += 2
        
        # Path 4: Donchian breakout + HMA bullish + 1w bullish
        if price_above_donchian and hma_bullish and price_above_1w_hma:
            long_score += 3
        
        # Path 5: Simple oversold (ensure trades in deep corrections)
        if crsi_extreme_low and not vol_extreme:
            long_score += 2
        
        # Path 6: HMA crossover bullish + RSI confirmation
        if hma_bullish and rsi_14[i] < 50 and crsi[i] < 35:
            long_score += 2
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 30:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_score += 3
        
        # Path 2: Range market + price above BB upper
        if is_range_market and price_above_bb_upper:
            short_score += 2
        
        # Path 3: Trend market + pullback + 1w bearish
        if is_trend_market and price_below_1w_hma and crsi[i] > 60:
            short_score += 2
        
        # Path 4: Donchian breakdown + HMA bearish + 1w bearish
        if price_below_donchian and hma_bearish and price_below_1w_hma:
            short_score += 3
        
        # Path 5: Simple overbought
        if crsi_extreme_high and not vol_extreme:
            short_score += 2
        
        # Path 6: HMA crossover bearish + RSI confirmation
        if hma_bearish and rsi_14[i] > 50 and crsi[i] > 65:
            short_score += 2
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if price_above_1w_hma and crsi[i] < 35:
                new_signal = current_size * 0.4
            elif price_below_1w_hma and crsi[i] > 65:
                new_signal = -current_size * 0.4
            elif crsi[i] < 20:
                new_signal = current_size * 0.3
            elif crsi[i] > 80:
                new_signal = -current_size * 0.3
        
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
            if position_side > 0 and is_trend_market and price_below_1w_hma and adx_14[i] > 30:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and price_above_1w_hma and adx_14[i] > 30:
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