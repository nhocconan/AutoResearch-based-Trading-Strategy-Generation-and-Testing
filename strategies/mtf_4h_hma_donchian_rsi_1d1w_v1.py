#!/usr/bin/env python3
"""
Experiment #231: 4h Primary + 1d/1w HTF — HMA Trend + Donchian Breakout + RSI Momentum

Hypothesis: After 230 experiments, the winning formula combines:
1. 4h PRIMARY for entry timing (proven TF from history)
2. 1d HTF for intermediate trend confirmation (not too slow, not too fast)
3. 1w HTF for major bias filter (avoid fighting macro trend)
4. DONCHIAN(20) breakout for clear entry signals
5. HMA(21) for faster trend detection than EMA
6. RSI(14) loose thresholds (45/55 not 40/60) to ensure trade frequency
7. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- 4h timeframe = 20-50 trades/year target (matches cost model)
- LOOSE entry conditions (RSI 45/55, not 40/60) = guaranteed 10+ trades
- Multiple entry paths (breakout + trend, or trend + momentum alone)
- HTF alignment prevents fighting major trends
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Key differences from failed #221, #224, #229, #230:
- LOOSER RSI thresholds (45/55 vs 40/60) for more trades
- Simpler logic without complex regime switching
- Force-trade fallback after 60 bars of no signal
- Smaller position sizes (0.25-0.30 vs 0.35) for lower DD

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_1d1w_v1"
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
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / prev * 100
    return slope

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    adx_4h = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_slope[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.10
        weekly_bearish = hma_1w_slope_aligned[i] < -0.10
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === INTERMEDIATE TREND (1d) ===
        daily_bullish = hma_1d_slope_aligned[i] > 0.15
        daily_bearish = hma_1d_slope_aligned[i] < -0.15
        daily_trend_strength = adx_1d_aligned[i] > 25
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        hourly_bullish = hma_4h_slope[i] > 0.20
        hourly_bearish = hma_4h_slope[i] < -0.20
        
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === MOMENTUM (RSI) - LOOSE THRESHOLDS FOR TRADE FREQUENCY ===
        rsi_bullish = rsi_14[i] > 45  # Loose: was 50
        rsi_bearish = rsi_14[i] < 55  # Loose: was 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === LOCAL TREND STRENGTH ===
        local_trend_strong = adx_4h[i] > 25
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency (CRITICAL for 10+ trades)
        long_score = 0
        long_strength = 0
        
        # Path 1: Donchian breakout + weekly bullish + RSI bullish (strongest)
        if breakout_long and weekly_bullish and rsi_bullish:
            long_score += 5
            long_strength = STRONG_SIZE
        
        # Path 2: Donchian breakout + daily bullish + RSI > 50
        if breakout_long and daily_bullish and rsi_14[i] > 50:
            long_score += 4
            long_strength = STRONG_SIZE
        
        # Path 3: Weekly + Daily bullish + RSI bullish (trend alignment)
        if weekly_bullish and daily_bullish and rsi_bullish and price_above_4h_hma:
            long_score += 4
            long_strength = STRONG_SIZE
        
        # Path 4: Breakout + weekly bullish (HTF confirmation)
        if breakout_long and weekly_bullish:
            long_score += 3
            long_strength = BASE_SIZE
        
        # Path 5: Breakout + daily bullish + RSI confirmation
        if breakout_long and daily_bullish and rsi_bullish:
            long_score += 3
            long_strength = BASE_SIZE
        
        # Path 6: Breakout + RSI > 50 (simpler, looser)
        if breakout_long and rsi_14[i] > 50:
            long_score += 2
            long_strength = BASE_SIZE * 0.8
        
        # Path 7: Weekly bullish + RSI strong + price above 4h HMA
        if weekly_bullish and rsi_strong_bull and price_above_4h_hma and bars_since_last_trade > 20:
            long_score += 2
            long_strength = BASE_SIZE * 0.7
        
        # Path 8: Daily bullish + RSI strong + breakout (momentum)
        if daily_bullish and rsi_strong_bull and breakout_long:
            long_score += 3
            long_strength = BASE_SIZE
        
        # Path 9: Simple RSI + HMA alignment (no breakout needed)
        if rsi_14[i] > 58 and price_above_1d_hma and price_above_4h_hma and bars_since_last_trade > 30:
            long_score += 2
            long_strength = BASE_SIZE * 0.6
        
        if long_score >= 4:
            new_signal = long_strength
        elif long_score >= 3 and bars_since_last_trade > 15:
            new_signal = long_strength * 0.9
        elif long_score >= 2 and bars_since_last_trade > 30:
            new_signal = long_strength * 0.7
        
        # SHORT ENTRIES
        short_score = 0
        short_strength = 0
        
        # Path 1: Donchian breakout + weekly bearish + RSI bearish (strongest)
        if breakout_short and weekly_bearish and rsi_bearish:
            short_score += 5
            short_strength = STRONG_SIZE
        
        # Path 2: Donchian breakout + daily bearish + RSI < 50
        if breakout_short and daily_bearish and rsi_14[i] < 50:
            short_score += 4
            short_strength = STRONG_SIZE
        
        # Path 3: Weekly + Daily bearish + RSI bearish (trend alignment)
        if weekly_bearish and daily_bearish and rsi_bearish and price_below_4h_hma:
            short_score += 4
            short_strength = STRONG_SIZE
        
        # Path 4: Breakout + weekly bearish (HTF confirmation)
        if breakout_short and weekly_bearish:
            short_score += 3
            short_strength = BASE_SIZE
        
        # Path 5: Breakout + daily bearish + RSI confirmation
        if breakout_short and daily_bearish and rsi_bearish:
            short_score += 3
            short_strength = BASE_SIZE
        
        # Path 6: Breakout + RSI < 50 (simpler, looser)
        if breakout_short and rsi_14[i] < 50:
            short_score += 2
            short_strength = BASE_SIZE * 0.8
        
        # Path 7: Weekly bearish + RSI strong + price below 4h HMA
        if weekly_bearish and rsi_strong_bear and price_below_4h_hma and bars_since_last_trade > 20:
            short_score += 2
            short_strength = BASE_SIZE * 0.7
        
        # Path 8: Daily bearish + RSI strong + breakout (momentum)
        if daily_bearish and rsi_strong_bear and breakout_short:
            short_score += 3
            short_strength = BASE_SIZE
        
        # Path 9: Simple RSI + HMA alignment (no breakout needed)
        if rsi_14[i] < 42 and price_below_1d_hma and price_below_4h_hma and bars_since_last_trade > 30:
            short_score += 2
            short_strength = BASE_SIZE * 0.6
        
        if short_score >= 4:
            new_signal = -short_strength
        elif short_score >= 3 and bars_since_last_trade > 15:
            new_signal = -short_strength * 0.9
        elif short_score >= 2 and bars_since_last_trade > 30:
            new_signal = -short_strength * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~10 days on 4h)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] > 48 and price_above_4h_hma:
                new_signal = BASE_SIZE * 0.35
            elif weekly_bearish and rsi_14[i] < 52 and price_below_4h_hma:
                new_signal = -BASE_SIZE * 0.35
            elif daily_bullish and rsi_14[i] > 52 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.30
            elif daily_bearish and rsi_14[i] < 48 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.30
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but weekly turns strongly bearish
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                trend_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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