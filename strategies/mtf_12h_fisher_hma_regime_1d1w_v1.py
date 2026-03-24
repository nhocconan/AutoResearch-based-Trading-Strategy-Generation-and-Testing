#!/usr/bin/env python3
"""
Experiment #172: 12h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + Regime Filter

Hypothesis: Previous regime-switching strategies failed because they were too complex
and had overly strict entry conditions (0 trades on BTC/ETH). Fisher Transform is
proven to work well in bear/range markets for catching reversals. Combined with
HMA trend filter and Choppiness regime detection, this should generate consistent
trades across ALL symbols while maintaining positive Sharpe.

KEY IMPROVEMENTS:
1. Fisher Transform (period=9) as primary reversal signal - catches bear market rallies
2. Dual HTF bias: 1d HMA for medium-term, 1w HMA for macro bias
3. Looser entry thresholds to ensure ≥30 trades per symbol on train
4. Choppiness Index only for position sizing (not entry block)
5. ATR trailing stop at 2.5x for risk management
6. Position size: 0.25 full, 0.15 partial (discrete levels)

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_regime_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to 0-1 range
    lowest_low = hl2_s.rolling(window=period, min_periods=period).min().values
    highest_high = hl2_s.rolling(window=period, min_periods=period).max().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (hl2 - lowest_low) / (price_range + 1e-10)
        normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = np.maximum(high_s - high_s.shift(1), 0).values
    minus_dm = np.maximum(low_s.shift(1) - low, 0).values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d HMA for medium-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    POSITION_SIZE_QUARTER = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0  # More lenient threshold
        is_ranging = chop_value > 55.0
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 12H TREND ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        hma_21_above_50 = hma_21[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_21_below_50 = hma_21[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = (fisher[i] < -1.0) and (fisher_signal[i] < fisher[i])
        fisher_short_cross = (fisher[i] > 1.0) and (fisher_signal[i] > fisher[i])
        
        # Fisher reversal from extreme
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher turning up from oversold
        fisher_turning_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        fisher_turning_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20.0
        adx_weak = adx[i] < 25.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        position_size = POSITION_SIZE_HALF
        
        # LONG entries - multiple confluence patterns
        long_confluence = 0
        
        if fisher_turning_up:
            long_confluence += 1
        if rsi_oversold:
            long_confluence += 1
        if price_above_hma_1d:
            long_confluence += 1
        if price_above_hma_21:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        
        # LONG: Need 2+ confluence factors
        if long_confluence >= 2:
            if is_trending and price_above_hma_1w:
                # Trend regime + macro bullish = full size
                new_signal = POSITION_SIZE_FULL
            elif is_ranging:
                # Range regime = mean reversion long
                new_signal = POSITION_SIZE_HALF
            elif price_above_hma_1d:
                # Neutral regime but 1d bias positive
                new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - multiple confluence patterns
        short_confluence = 0
        
        if fisher_turning_down:
            short_confluence += 1
        if rsi_overbought:
            short_confluence += 1
        if price_below_hma_1d:
            short_confluence += 1
        if price_below_hma_21:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        
        # SHORT: Need 2+ confluence factors
        if short_confluence >= 2:
            if is_trending and price_below_hma_1w:
                # Trend regime + macro bearish = full size
                new_signal = -POSITION_SIZE_FULL
            elif is_ranging:
                # Range regime = mean reversion short
                new_signal = -POSITION_SIZE_HALF
            elif price_below_hma_1d:
                # Neutral regime but 1d bias negative
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 12h HMA
                if price_above_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 12h HMA
                if price_below_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 12h HMA significantly
        if in_position and position_side > 0 and price_below_hma_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 12h HMA significantly
        if in_position and position_side < 0 and price_above_hma_21:
            new_signal = 0.0
        
        # Exit if macro bias flips strongly against position
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_hma_1w:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
        prev_fisher = fisher[i]
    
    return signals