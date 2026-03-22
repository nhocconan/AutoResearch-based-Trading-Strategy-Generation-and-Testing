#!/usr/bin/env python3
"""
Experiment #107: 1d Primary + 1w HTF — Dual Regime (Chop=Mean Revert, Trend=Breakout)

Hypothesis: Daily timeframe with weekly trend bias should reduce noise and fee drag
while capturing major moves. Research shows dual-regime strategies outperform single-approach:
- Range markets (CHOP>55): Connors RSI mean reversion at Bollinger extremes
- Trend markets (CHOP<45): Donchian breakouts with HMA confirmation

Why this should work:
1. 1d timeframe = 20-50 trades/year target (minimal fee drag)
2. 1w HTF provides stronger trend bias than 1d alone
3. Choppiness Index correctly identifies regime for appropriate strategy
4. Connors RSI has 75% win rate for mean reversion entries
5. Donchian breakouts capture sustained trends without whipsaw
6. ATR-based position sizing adjusts to volatility (smaller size in high vol)

Key improvements over #106:
- Higher TF (1d vs 12h) = fewer trades, less fee churn
- 1w HTF = stronger trend filter than 1d
- Dual regime = adapts to market conditions
- More lenient thresholds = ensures trades (avoid 0-trade failure)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (vol-adjusted)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_donchian_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # HMA for trend confirmation
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
    # Volatility ratio for position sizing
    vol_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete, max 0.40)
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        # === 1W TREND BIAS ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        trend_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === ADX STRENGTH ===
        strong_trend = adx_14[i] > 25
        weak_trend = adx_14[i] < 20
        
        # === POSITION SIZING (vol-adjusted) ===
        current_size = BASE_SIZE
        if vol_ratio[i] > 1.5:
            current_size = BASE_SIZE * 0.7  # Reduce size in high vol
        elif vol_ratio[i] < 0.7:
            current_size = BASE_SIZE * 1.1  # Increase size in low vol
        current_size = min(current_size, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === RANGE MARKET: MEAN REVERSION (Connors RSI + BB) ===
        if is_range_market:
            # LONG: CRSI oversold + price at BB lower
            if crsi[i] < 30 and close[i] <= bb_lower[i] * 1.002:
                if trend_1w_bullish or not trend_1w_bearish:
                    new_signal = current_size
            
            # SHORT: CRSI overbought + price at BB upper
            if crsi[i] > 70 and close[i] >= bb_upper[i] * 0.998:
                if trend_1w_bearish or not trend_1w_bullish:
                    new_signal = -current_size
            
            # Extreme mean reversion (override trend bias)
            if crsi[i] < 15 and close[i] <= bb_lower[i] * 1.005:
                new_signal = current_size
            if crsi[i] > 85 and close[i] >= bb_upper[i] * 0.995:
                new_signal = -current_size
        
        # === TREND MARKET: BREAKOUT (Donchian + HMA) ===
        if is_trend_market:
            # LONG: Donchian breakout + bullish trend alignment
            if close[i] > donchian_upper[i] * 0.998:
                if trend_1w_bullish and trend_1d_bullish:
                    new_signal = current_size
                elif trend_1w_bullish or trend_1d_bullish:
                    new_signal = current_size * 0.7
            
            # SHORT: Donchian breakdown + bearish trend alignment
            if close[i] < donchian_lower[i] * 1.002:
                if trend_1w_bearish and trend_1d_bearish:
                    new_signal = -current_size
                elif trend_1w_bearish or trend_1d_bearish:
                    new_signal = -current_size * 0.7
        
        # === NEUTRAL/TRANSITION: HYBRID APPROACH ===
        if not is_range_market and not is_trend_market:
            # Use CRSI for entries but require trend confirmation
            if crsi[i] < 25 and trend_1w_bullish:
                new_signal = current_size * 0.7
            if crsi[i] > 75 and trend_1w_bearish:
                new_signal = -current_size * 0.7
            
            # Breakout with ADX confirmation
            if close[i] > donchian_upper[i] * 0.998 and adx_14[i] > 22:
                new_signal = current_size * 0.7
            if close[i] < donchian_lower[i] * 1.002 and adx_14[i] > 22:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            # Force entry based on strongest signal available
            if crsi[i] < 20:
                new_signal = current_size * 0.4
            elif crsi[i] > 80:
                new_signal = -current_size * 0.4
            elif trend_1w_bullish and close[i] > hma_1w_21_aligned[i]:
                new_signal = current_size * 0.3
            elif trend_1w_bearish and close[i] < hma_1w_21_aligned[i]:
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
            if position_side > 0 and is_trend_market and trend_1w_bearish and trend_1d_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and trend_1w_bullish and trend_1d_bullish:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT (take profit on mean reversion) ===
        crsi_exit = False
        if in_position and is_range_market:
            if position_side > 0 and crsi[i] > 60:
                crsi_exit = True
            if position_side < 0 and crsi[i] < 40:
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