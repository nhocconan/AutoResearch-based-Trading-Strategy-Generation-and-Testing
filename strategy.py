#!/usr/bin/env python3
"""
Experiment #137: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 1d strategies failed because they were either too strict (0 trades)
or too simple (single indicator). This strategy combines:

1. REGIME DETECTION: Choppiness Index switches between mean-revert and trend-follow
2. 1W HMA TREND: Major bias from weekly timeframe (stronger than 1d)
3. DUAL ENTRY PATHS: 
   - Range regime: RSI extremes + Bollinger bands (mean reversion)
   - Trend regime: Donchian breakout + HMA confirmation (trend follow)
4. VOLATILITY FILTER: ATR ratio ensures we enter on meaningful moves
5. LENIENT THRESHOLDS: Multiple entry paths ensure we get 20-50 trades/year

Why this should work:
- 1d timeframe = natural trade frequency (20-50/year target)
- 1w HTF = stronger trend filter than 1d/4h used in failed strategies
- Dual regime = adapts to market conditions (range vs trend)
- Multiple entry paths = ensures trades on all symbols (BTC/ETH/SOL)
- Conservative sizing (0.25-0.30) = controls drawdown in 2022 crash

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol (80-200 over 4yr train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_donchian_rsi_1w_v1"
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
    """Calculate Donchian Channel (highest high / lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_hma_slope(hma_values, lookback=3):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # 1d HMA for trend confirmation
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_48 = calculate_hma(close, 48)
    
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
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W TREND BIAS (stronger filter) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = hma_1d_21[i] > hma_1d_48[i]
        trend_1d_bearish = hma_1d_21[i] < hma_1d_48[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === VOLATILITY FILTER ===
        vol_normal = atr_ratio[i] > 0.8  # Not in extreme vol crush
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        rsi_7_low = rsi_7[i] < 35
        rsi_7_high = rsi_7[i] > 65
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        
        # === DONCHIAN BREAKOUT ===
        price_above_donchian = close[i] > donchian_upper[i]
        price_below_donchian = close[i] < donchian_lower[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (ensure we get trades)
        long_score = 0
        
        # Path 1: Range market + RSI oversold + BB lower (mean reversion)
        if is_range_market and rsi_oversold and price_below_bb_lower:
            long_score += 3
        
        # Path 2: Range market + RSI extreme low
        if is_range_market and rsi_extreme_low:
            long_score += 2
        
        # Path 3: Trend market + 1w bullish + pullback (trend follow)
        if is_trend_market and trend_1w_bullish and rsi_7_low:
            long_score += 2
        
        # Path 4: Trend market + Donchian breakout + 1w bullish
        if is_trend_market and price_above_donchian and trend_1w_bullish:
            long_score += 3
        
        # Path 5: Price above 1w HMA + RSI pullback
        if price_above_1w_hma and rsi_14[i] < 45:
            long_score += 2
        
        # Path 6: Simple RSI extreme (fallback for more trades)
        if rsi_14[i] < 28:
            long_score += 1
        
        # Path 7: BB squeeze breakout long
        if bb_width < 0.05 and price_above_bb_upper and trend_1w_bullish:
            long_score += 2
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + RSI overbought + BB upper
        if is_range_market and rsi_overbought and price_above_bb_upper:
            short_score += 3
        
        # Path 2: Range market + RSI extreme high
        if is_range_market and rsi_extreme_high:
            short_score += 2
        
        # Path 3: Trend market + 1w bearish + pullback
        if is_trend_market and trend_1w_bearish and rsi_7_high:
            short_score += 2
        
        # Path 4: Trend market + Donchian breakdown + 1w bearish
        if is_trend_market and price_below_donchian and trend_1w_bearish:
            short_score += 3
        
        # Path 5: Price below 1w HMA + RSI rally
        if price_below_1w_hma and rsi_14[i] > 55:
            short_score += 2
        
        # Path 6: Simple RSI extreme (fallback)
        if rsi_14[i] > 72:
            short_score += 1
        
        # Path 7: BB squeeze breakout short
        if bb_width < 0.05 and price_below_bb_lower and trend_1w_bearish:
            short_score += 2
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD (ensure trades) ===
        # Force trade if no signal for 90 bars (~90 days on 1d)
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.5
            elif rsi_14[i] < 25:
                new_signal = current_size * 0.4
            elif rsi_14[i] > 75:
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
            # Exit long if regime switches to strong trend bearish
            if position_side > 0 and is_trend_market and trend_1w_bearish:
                regime_reversal = True
            # Exit short if regime switches to strong trend bullish
            if position_side < 0 and is_trend_market and trend_1w_bullish:
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