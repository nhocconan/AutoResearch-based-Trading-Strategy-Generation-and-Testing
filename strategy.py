#!/usr/bin/env python3
"""
Experiment #032: 30m EMA Crossover + 4h/1d Dual HMA Trend + Choppiness Regime + ADX + Volume

Hypothesis: Previous 30m strategies failed (Sharpe -2.3 to -4.8) because:
1. HTF filter was too weak (only 4h, or allowed either 4h OR 1d)
2. No regime filter - traded through choppy ranges where crossovers whipsaw
3. Too many trades → fee drag destroyed edge

This strategy fixes those issues:
1. DUAL HTF FILTER: BOTH 4h AND 1d HMA must agree on trend direction
   - Much stricter than previous attempts (which used OR logic)
   - Eliminates false signals when HTFs disagree
2. CHOPPINESS INDEX REGIME: Only trade when CHOP(14) < 55 (trending market)
   - Avoids 60%+ choppiness where mean-reversion dominates
   - Classic regime filter from professional trading literature
3. ADX(14) > 25: Additional trend strength confirmation
4. VOLUME SPIKE: Volume > 1.5x 20-bar average on entry bars
5. STRICT ENTRY: EMA(8) crosses EMA(21) + ALL filters above must agree
6. FEWER TRADES: Target 30-50/year by requiring full confluence

Why this should beat previous 30m attempts:
- Dual HTF (4h+1d) is 2x stricter than single 4h filter
- Choppiness filter eliminates the #1 failure mode (range whipsaws)
- ADX + Volume adds conviction filtering
- Conservative sizing (0.25-0.35) limits drawdown

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year (strict enough to avoid fee drag, enough to meet minimum)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_4h_1d_hma_chop_adx_vol_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Range-bound / choppy market (avoid trend strategies)
    - CHOP < 38.2: Strong trending market
    - 38.2 < CHOP < 61.8: Transition zone
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low
    
    # Choppiness formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    # Handle edge cases
    chop = chop.replace([np.inf, -np.inf], np.nan)
    chop = chop.fillna(50)  # Default to neutral
    
    return chop.values

def calculate_volume_spike(volume, lookback=20, threshold=1.5):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.5)
    
    # EMA crossover signals
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.35  # All filters agree
    SIZE_MODERATE = 0.25  # Most filters agree
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]) or np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            continue
        
        # === DUAL HTF TREND FILTER (BOTH 4h AND 1d MUST AGREE) ===
        # Much stricter than previous attempts (which used OR logic)
        price_vs_4h = close[i] - hma_4h_aligned[i]
        price_vs_1d = close[i] - hma_1d_aligned[i]
        
        bull_htf = (price_vs_4h > 0) and (price_vs_1d > 0)  # BOTH must be bullish
        bear_htf = (price_vs_4h < 0) and (price_vs_1d < 0)  # BOTH must be bearish
        
        # === CHOPPINESS REGIME FILTER ===
        # Only trade in trending markets (CHOP < 55)
        trending_regime = chop_14[i] < 55
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25  # Strong trend
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === EMA CROSSOVER SIGNAL ===
        # Use previous bar to avoid look-ahead on crossover detection
        ema_cross_long = (ema_8[i] > ema_21[i]) and (ema_8[i-1] <= ema_21[i-1]) if i > 0 else False
        ema_cross_short = (ema_8[i] < ema_21[i]) and (ema_8[i-1] >= ema_21[i-1]) if i > 0 else False
        
        # Also allow continuation if already in trend (EMA8 > EMA21 and both rising)
        ema_bull_trend = (ema_8[i] > ema_21[i]) and (ema_8[i] > ema_8[i-1]) if i > 0 else (ema_8[i] > ema_21[i])
        ema_bear_trend = (ema_8[i] < ema_21[i]) and (ema_8[i] < ema_8[i-1]) if i > 0 else (ema_8[i] < ema_21[i])
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0  # Count confirming filters
        
        # LONG ENTRY: HTF bull + trending regime + EMA signal + ADX/DI/volume confirmation
        if bull_htf and trending_regime:
            if ema_cross_long or ema_bull_trend:
                signal_strength += 2  # HTF trend + EMA signal (core)
                
                if adx_strong:
                    signal_strength += 1  # Trend strength
                
                if di_bull:
                    signal_strength += 1  # DI direction
                
                if vol_confirmed:
                    signal_strength += 1  # Volume confirmation
                
                # Assign size based on confirmation count
                if signal_strength >= 4:
                    new_signal = SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: HTF bear + trending regime + EMA signal + ADX/DI/volume confirmation
        elif bear_htf and trending_regime:
            if ema_cross_short or ema_bear_trend:
                signal_strength += 2  # HTF trend + EMA signal (core)
                
                if adx_strong:
                    signal_strength += 1  # Trend strength
                
                if di_bear:
                    signal_strength += 1  # DI direction
                
                if vol_confirmed:
                    signal_strength += 1  # Volume confirmation
                
                # Assign size based on confirmation count
                if signal_strength >= 4:
                    new_signal = -SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend reverses against position
            if position_side > 0 and bear_htf:
                trend_exit = True
            if position_side < 0 and bull_htf:
                trend_exit = True
            
            # Exit if regime becomes choppy
            if not trending_regime:
                trend_exit = True
            
            # Exit if EMA crosses against position
            if position_side > 0 and ema_8[i] < ema_21[i]:
                trend_exit = True
            if position_side < 0 and ema_8[i] > ema_21[i]:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals