#!/usr/bin/env python3
"""
Experiment #156: 12h Primary + 1d HTF — Regime-Adaptive ADX/CHOP Strategy

Hypothesis: Previous vol-spike strategies failed because they only triggered during
extreme events (too rare). This strategy uses REGIME-ADAPTIVE logic:

1. TRENDING REGIME (ADX>25, CHOP<45): Pullback entries to EMA21 with HMA trend bias
2. RANGING REGIME (ADX<20, CHOP>55): Mean reversion at Bollinger extremes
3. TRANSITION REGIME: Reduced position size, wait for clarity

Why this should work:
- ADX filters out whipsaw periods (major issue in 2022 BTC crash)
- CHOP confirms regime (range vs trend)
- Different logic per regime = adapts to market conditions
- 1d HMA slope prevents counter-trend trades in strong moves
- More lenient thresholds = ensures minimum trade count

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adx_chop_1d_v1"
timeframe = "12h"
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
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_pct = 100 * plus_di / np.where(atr > 0, atr, 1e-10)
    minus_di_pct = 100 * minus_di / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di_pct - minus_di_pct) / np.where((plus_di_pct + minus_di_pct) > 0, (plus_di_pct + minus_di_pct), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.30
    RANGE_SIZE = 0.25
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope = (hma_1d_21_aligned[i] - hma_1d_21_aligned[i-10]) / hma_1d_21_aligned[i-10] * 100 if i > 10 and hma_1d_21_aligned[i-10] != 0 else 0
        trend_1d_bullish = hma_1d_slope > 0.2
        trend_1d_bearish = hma_1d_slope < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === REGIME CLASSIFICATION ===
        is_trending = adx_14[i] > 25 and chop_14[i] < 45
        is_ranging = adx_14[i] < 20 and chop_14[i] > 55
        is_transition = not is_trending and not is_ranging
        
        # === POSITION SIZE BY REGIME ===
        current_size = BASE_SIZE
        if is_trending:
            current_size = TREND_SIZE
        elif is_ranging:
            current_size = RANGE_SIZE
        else:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_conditions = 0
        
        # Trending regime: pullback to EMA21 + 1d bullish
        if is_trending and trend_1d_bullish:
            if close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.98:
                long_conditions += 2
            if rsi_14[i] < 45:
                long_conditions += 1
            if price_above_1d_hma:
                long_conditions += 1
        
        # Ranging regime: mean reversion at BB lower
        if is_ranging:
            if close[i] <= bb_lower[i] * 1.005:
                long_conditions += 2
            if rsi_14[i] < 40:
                long_conditions += 2
            if rsi_14[i] < 30:
                long_conditions += 1
        
        # Transition regime: wait for stronger signals
        if is_transition:
            if close[i] <= bb_lower[i] * 0.99 and rsi_14[i] < 35:
                long_conditions += 2
            if trend_1d_bullish and rsi_14[i] < 40:
                long_conditions += 1
        
        # General oversold (ensure minimum trades)
        if rsi_14[i] < 32 and close[i] <= bb_lower[i] * 1.01:
            long_conditions += 1
        
        if long_conditions >= 3:
            new_signal = current_size
        elif long_conditions == 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Trending regime: pullback to EMA21 + 1d bearish
        if is_trending and trend_1d_bearish:
            if close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.02:
                short_conditions += 2
            if rsi_14[i] > 55:
                short_conditions += 1
            if price_below_1d_hma:
                short_conditions += 1
        
        # Ranging regime: mean reversion at BB upper
        if is_ranging:
            if close[i] >= bb_upper[i] * 0.995:
                short_conditions += 2
            if rsi_14[i] > 60:
                short_conditions += 2
            if rsi_14[i] > 70:
                short_conditions += 1
        
        # Transition regime: wait for stronger signals
        if is_transition:
            if close[i] >= bb_upper[i] * 1.01 and rsi_14[i] > 65:
                short_conditions += 2
            if trend_1d_bearish and rsi_14[i] > 60:
                short_conditions += 1
        
        # General overbought (ensure minimum trades)
        if rsi_14[i] > 68 and close[i] >= bb_upper[i] * 0.99:
            short_conditions += 1
        
        if short_conditions >= 3:
            new_signal = -current_size
        elif short_conditions == 2 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 70:
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
            # Exit long if regime shifts to strong bearish trend
            if position_side > 0 and is_trending and trend_1d_bearish and adx_14[i] > 30:
                regime_reversal = True
            # Exit short if regime shifts to strong bullish trend
            if position_side < 0 and is_trending and trend_1d_bullish and adx_14[i] > 30:
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