#!/usr/bin/env python3
"""
Experiment #015: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime

Hypothesis: 1h timeframe with Ehlers Fisher Transform for entries + Choppiness regime
filter + 4h HMA trend direction will generate 40-80 trades/year with positive Sharpe.

Why this should work (learning from 14 failed experiments):
1. Fisher Transform catches reversals better than RSI in bear/range markets (BTC 2025)
2. Choppiness Index properly distinguishes trend vs range regimes
3. 4h HMA provides trend bias without being too slow (unlike 1w)
4. LOOSE entry thresholds ensure trade generation (Fisher > -1.8, not > -1.0)
5. Position size 0.25 (smaller for 1h to reduce fee drag from more trades)
6. Hold logic prevents churn on minor pullbacks

Key components:
1. Ehlers Fisher Transform (period=9): Normalizes price, catches reversals
2. Choppiness Index (14): Regime detection (CHOP>55=range, CHOP<45=trend)
3. 4h HMA(21): Trend direction bias
4. 1d ADX(14): Trend strength confirmation (ADX>20=trending)
5. ATR(14) stoploss: 2.0*ATR trailing stop

Position size: 0.25 (discrete, smaller for 1h timeframe)
Stoploss: 2.0*ATR trailing
Target trades: 40-80/year (use loose entries, strict HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_regime_4h1d_v1"
timeframe = "1h"
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
    Transforms price into a Gaussian normal distribution for better reversal signals.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag of Fisher)
        if i > 0:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
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
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ADX for trend strength
    adx_1d, _, _ = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 1h HMA for local trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss and hold logic
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND STRENGTH ===
        adx_value = adx_1d_aligned[i]
        is_trending_1d = adx_value > 18.0  # LOOSE threshold for more trades
        is_ranging_1d = adx_value <= 18.0
        
        # === CHOPPINESS REGIME (1h) ===
        chop_value = chop_14[i]
        is_ranging_chop = chop_value > 52.0  # LOOSE for more trades
        is_trending_chop = chop_value < 48.0  # LOOSE for more trades
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds) ===
        fisher_oversold = fisher[i] < -1.5  # Was -1.0, now -1.5 for more longs
        fisher_overbought = fisher[i] > 1.5  # Was 1.0, now 1.5 for more shorts
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] <= -1.0
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] >= 1.0
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_oversold = rsi_14[i] < 40.0  # LOOSE for more trades
        rsi_overbought = rsi_14[i] > 60.0  # LOOSE for more trades
        
        # === VOLUME FILTER (LOOSE) ===
        vol_ok = vol_ratio[i] > 0.6  # Very loose - just not extremely low volume
        
        # === LOCAL TREND (1h HMA) ===
        price_above_hma_1h = close[i] > hma_1h[i]
        price_below_hma_1h = close[i] < hma_1h[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher ---
        if is_ranging_chop or is_ranging_1d:
            # Long: Fisher oversold + RSI confirmation + 4h bias helps
            if fisher_oversold or fisher_cross_up:
                if (rsi_oversold or price_above_hma_4h) and vol_ok:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + RSI confirmation + 4h bias helps
            elif fisher_overbought or fisher_cross_down:
                if (rsi_overbought or price_below_hma_4h) and vol_ok:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Fisher pullback ---
        elif is_trending_chop and is_trending_1d:
            # Long: 4h bullish + Fisher pulling back from oversold + 1h confirms
            if price_above_hma_4h:
                if (fisher[i] > -1.8 and fisher[i] < 0.0) and price_above_hma_1h:
                    if rsi_14[i] < 55:  # Not overbought
                        new_signal = POSITION_SIZE
            
            # Short: 4h bearish + Fisher pulling back from overbought + 1h confirms
            elif price_below_hma_4h:
                if (fisher[i] < 1.8 and fisher[i] > 0.0) and price_below_hma_1h:
                    if rsi_14[i] > 45:  # Not oversold
                        new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple Fisher crossover if no regime signal ---
        if new_signal == 0.0:
            # Long: Fisher crosses up from deep oversold
            if fisher_cross_up and fisher_signal[i] < -1.5:
                if vol_ok and price_above_hma_4h:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher crosses down from deep overbought
            elif fisher_cross_down and fisher_signal[i] > 1.5:
                if vol_ok and price_below_hma_4h:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (prevent churn on minor pullbacks) ===
        if in_position and new_signal == 0.0:
            # Hold long if Fisher not extremely overbought
            if position_side > 0 and fisher[i] < 2.5:
                new_signal = signals[i-1] if i > 0 else 0.0
            # Hold short if Fisher not extremely oversold
            elif position_side < 0 and fisher[i] > -2.5:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON STRONG REGIME CHANGE ===
        # Exit long if 4h trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and fisher[i] > 2.0:
                new_signal = 0.0
        
        # Exit short if 4h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and fisher[i] < -2.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_fisher = 0.0
        
        signals[i] = new_signal
    
    return signals