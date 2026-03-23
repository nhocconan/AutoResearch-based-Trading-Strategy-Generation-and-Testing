#!/usr/bin/env python3
"""
Experiment #028: 30m Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + Triple HTF

Hypothesis: 30m timeframe with Ehlers Fisher Transform for entry timing, Choppiness Index for
regime detection, and TRIPLE HTF confirmation (4h HMA trend + 1d ADX strength + 1d trend) will
generate 30-80 trades/year with positive Sharpe. Key insight: Fisher Transform catches reversals
in bear/range markets better than RSI, while triple HTF filter ensures we only trade with
macro trend alignment.

Strategy Logic:
1. FISHER TRANSFORM (30m): Entry trigger when Fisher crosses -1.5 (long) or +1.5 (short)
2. CHOPPINESS INDEX (30m): CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA (21): Macro trend bias — only long if price > 4h HMA, only short if price < 4h HMA
4. 1d ADX (14): Trend strength filter — ADX > 20 for trend regime, ADX < 25 for range regime
5. 1d HMA (21): Secondary trend confirmation — aligns with 4h HMA direction
6. SESSION FILTER: Only trade 8-20 UTC (high liquidity hours)
7. VOLUME FILTER: Volume > 0.8x 20-bar average
8. ATR(14) stoploss: 2.5*ATR trailing stop

Why this should work:
- 30m primary = enough signals for trade generation but not excessive
- Fisher Transform = proven reversal indicator in bear markets (better than RSI)
- Triple HTF (4h HMA + 1d ADX + 1d HMA) = strong trend alignment filter
- Choppiness regime = adapts entry logic to market conditions
- Session/volume filters = avoids low-liquidity whipsaws
- Smaller position size (0.25) = appropriate for lower TF fee sensitivity

Position size: 0.25 (smaller for 30m due to higher trade frequency)
Stoploss: 2.5*ATR trailing
Target trades: 40-80/year (strict filters to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_chop_triple_htf_session_v1"
timeframe = "30m"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.33
    """
    n = period
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(typical).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(typical).rolling(window=n, min_periods=n).min().values
    
    # Calculate X value (normalized price)
    price_range = hh - ll + 1e-10
    x = 0.67 * (typical - ll) / price_range - 0.33
    
    # Clamp X to avoid division by zero
    x = np.clip(x, -0.999, 0.999)
    
    # Calculate Fisher Transform
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
    
    # Calculate signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate DM+ and DM-
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, DM+, DM- using Wilder's method (EMA with span=n)
    tr_smooth = pd.Series(tr).ewm(span=n, min_periods=n, adjust=False).mean().values
    dmp_smooth = pd.Series(dm_plus).ewm(span=n, min_periods=n, adjust=False).mean().values
    dmm_smooth = pd.Series(dm_minus).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100.0 * dmp_smooth / (tr_smooth + 1e-10)
    di_minus = 100.0 * dmm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    di_sum = di_plus + di_minus + 1e-10
    dx = 100.0 * np.abs(di_plus - di_minus) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for secondary trend confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d ADX for trend strength
    adx_1d_raw = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_14[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H MACRO TREND ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1D ADX TREND STRENGTH ===
        adx_value = adx_1d_aligned[i]
        is_trending_1d = adx_value > 20.0  # Trending market
        is_ranging_1d = adx_value < 25.0   # Range market (with hysteresis)
        
        # === 30M CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging_30m = chop_value > 55.0  # Range market
        is_trending_30m = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # Fisher turning up/down
        fisher_rising = fisher[i] > fisher_signal[i]
        fisher_falling = fisher[i] < fisher_signal[i]
        
        # === COMBINED REGIME LOGIC ===
        new_signal = 0.0
        
        # Only trade during session with volume
        if not (is_session and volume_ok):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # --- RANGING REGIME: Mean Reversion with Fisher ---
        if is_ranging_30m or is_ranging_1d:
            # Long: Fisher oversold + price above 4h HMA (bullish bias)
            if fisher_oversold or (fisher_cross_up and fisher[i] < 0):
                if price_above_hma_4h and price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + price below 4h HMA (bearish bias)
            elif fisher_overbought or (fisher_cross_down and fisher[i] > 0):
                if price_below_hma_4h and price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Fisher ---
        elif is_trending_30m or is_trending_1d:
            # Long: Fisher crosses up from oversold + all HTF bullish
            if fisher_cross_up:
                if price_above_hma_4h and price_above_hma_1d and fisher_rising:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher crosses down from overbought + all HTF bearish
            elif fisher_cross_down:
                if price_below_hma_4h and price_below_hma_1d and fisher_falling:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple Fisher reversal if regime unclear ---
        if new_signal == 0.0:
            # Long: Fisher deeply oversold + HTF alignment
            if fisher[i] < -2.0 and fisher_rising:
                if price_above_hma_4h:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher deeply overbought + HTF alignment
            elif fisher[i] > 2.0 and fisher_falling:
                if price_below_hma_4h:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and price_above_hma_1d:
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
    
    return signals