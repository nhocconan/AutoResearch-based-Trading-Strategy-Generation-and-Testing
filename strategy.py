#!/usr/bin/env python3
"""
Experiment #129: 4h Primary + 1d HTF — Dual Regime Fisher + KAMA Adaptive

Hypothesis: Previous 4h strategies failed because they used单一 approach (either pure trend or pure mean revert). 
Research shows crypto markets alternate between trending and ranging regimes. This strategy uses:

1. REGIME DETECTION: Choppiness Index + ADX to classify market state
   - CHOP > 61.8 + ADX < 20 = Range (mean revert)
   - CHOP < 38.2 + ADX > 25 = Trend (trend follow)
   - Otherwise = Transition (reduced position)

2. ENTRY TIMING: Ehlers Fisher Transform for reversals (better than RSI in bear markets)
   - Fisher crosses above -1.5 from below = Long signal
   - Fisher crosses below +1.5 from above = Short signal

3. TREND FILTER: 1d HMA(21) slope for major trend bias
   - Only long if 1d HMA slope > 0 (or deep oversold in bear)
   - Only short if 1d HMA slope < 0 (or deep overbought in bull)

4. ADAPTIVE TREND: KAMA(14) adjusts to market efficiency ratio
   - KAMA rises faster in trends, flattens in ranges

5. VOLATILITY FILTER: ATR ratio to avoid entries during extreme vol spikes

Why this should work:
- Fisher Transform catches reversals better than RSI (proven in bear markets)
- KAMA adapts to changing market conditions (less whipsaw than EMA/HMA)
- Dual regime = right tool for right market condition
- 4h timeframe = 20-50 trades/year target (low fee drag)
- 1d HTF prevents fighting major trends

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_dual_regime_1d_v1"
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
    KAMA adapts to market noise via Efficiency Ratio.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    signal = np.abs(close_s.diff(er_period))
    noise = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = signal / np.where(noise > 0, noise, 1e-10)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for better reversal signals.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price
    median = (high + low) / 2.0
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(median).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(median).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    normalized = 2.0 * (median - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log(0)
    
    # Fisher Transform
    for i in range(period, n):
        fisher[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i]))
        if i > 0:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    kama_14 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    RANGE_SIZE = 0.25
    TREND_SIZE = 0.30
    TRANSITION_SIZE = 0.20
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(kama_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === REGIME DETECTION ===
        is_range_market = chop_14[i] > 58 and adx_14[i] < 22
        is_trend_market = chop_14[i] < 42 and adx_14[i] > 25
        is_transition = not is_range_market and not is_trend_market
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_14[i] and kama_14[i] > kama_14[i-1] if i > 0 else False
        kama_bearish = close[i] < kama_14[i] and kama_14[i] < kama_14[i-1] if i > 0 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_signal = fisher[i] > -1.5 and fisher_trigger[i] < -1.5 if i > 0 else False
        fisher_short_signal = fisher[i] < 1.5 and fisher_trigger[i] > 1.5 if i > 0 else False
        
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === VOLATILITY FILTER ===
        vol_extreme = atr_ratio[i] > 2.0  # Avoid entries during extreme vol
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        if is_range_market:
            current_size = RANGE_SIZE
        elif is_trend_market:
            current_size = TREND_SIZE
        else:
            current_size = TRANSITION_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        
        # Path 1: Range market + Fisher extreme low + BB lower (mean revert)
        if is_range_market and fisher_extreme_low and price_below_bb_lower:
            long_score += 4
        
        # Path 2: Range market + Fisher long signal + RSI oversold
        if is_range_market and fisher_long_signal and rsi_oversold:
            long_score += 3
        
        # Path 3: Trend market + 1d bullish + KAMA bullish + Fisher long
        if is_trend_market and trend_1d_bullish and kama_bullish and fisher_long_signal:
            long_score += 3
        
        # Path 4: Trend market + pullback to KAMA + 1d bullish
        if is_trend_market and trend_1d_bullish and close[i] < kama_14[i] * 1.01 and close[i] > kama_14[i] * 0.98:
            if fisher_long_signal or rsi_oversold:
                long_score += 2
        
        # Path 5: 1d bullish + deep oversold (counter-trend pullback in bull)
        if trend_1d_bullish and rsi_extreme_low and fisher_extreme_low:
            long_score += 3
        
        # Path 6: Transition + strong Fisher + RSI confluence
        if is_transition and fisher_extreme_low and rsi_oversold and price_below_bb_lower:
            long_score += 2
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + Fisher extreme high + BB upper
        if is_range_market and fisher_extreme_high and price_above_bb_upper:
            short_score += 4
        
        # Path 2: Range market + Fisher short signal + RSI overbought
        if is_range_market and fisher_short_signal and rsi_overbought:
            short_score += 3
        
        # Path 3: Trend market + 1d bearish + KAMA bearish + Fisher short
        if is_trend_market and trend_1d_bearish and kama_bearish and fisher_short_signal:
            short_score += 3
        
        # Path 4: Trend market + pullback to KAMA + 1d bearish
        if is_trend_market and trend_1d_bearish and close[i] > kama_14[i] * 0.99 and close[i] < kama_14[i] * 1.02:
            if fisher_short_signal or rsi_overbought:
                short_score += 2
        
        # Path 5: 1d bearish + deep overbought (counter-trend rally in bear)
        if trend_1d_bearish and rsi_extreme_high and fisher_extreme_high:
            short_score += 3
        
        # Path 6: Transition + strong Fisher + RSI confluence
        if is_transition and fisher_extreme_high and rsi_overbought and price_above_bb_upper:
            short_score += 2
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~33 days on 4h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and (rsi_oversold or fisher_extreme_low):
                new_signal = current_size * 0.5
            elif trend_1d_bearish and (rsi_overbought or fisher_extreme_high):
                new_signal = -current_size * 0.5
            elif is_range_market and fisher_extreme_low:
                new_signal = current_size * 0.4
            elif is_range_market and fisher_extreme_high:
                new_signal = -current_size * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:  # Fisher overbought
                fisher_exit = True
            if position_side < 0 and fisher[i] < -1.5:  # Fisher oversold
                fisher_exit = True
        
        if stoploss_triggered or regime_reversal or fisher_exit:
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