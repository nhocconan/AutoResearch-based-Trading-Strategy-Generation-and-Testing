#!/usr/bin/env python3
"""
Experiment #301: 15m EMA Crossover with Dual HTF HMA Bias and Volume Confirmation

Hypothesis: After analyzing 298+ experiments, clear patterns emerge:
1. 15m timeframe has FAILED catastrophically in past attempts (#289 Sharpe=-3.234, #295 Sharpe=-2.911)
2. The issue is NOT the indicator - it's excessive noise and fee drag at 15m
3. 4h Supertrend + 1d HMA works best (#292 Sharpe=0.485) - proves HTF bias is critical
4. Mean reversion ALWAYS fails across all timeframes
5. Simple trend following with STRONG HTF filter is the only winning formula

This strategy adapts the winning formula for 15m:
1. 4h HMA(21) for PRIMARY directional bias (proven edge from #292)
2. 1h HMA(21) for SECONDARY confirmation (stricter filter than past 15m attempts)
3. EMA(8)/EMA(21) crossover on 15m for entry timing (simpler than Supertrend)
4. ADX(14) > 20 for trend strength (higher threshold to reduce whipsaws)
5. Volume confirmation: taker_buy_volume ratio > 1.1 for longs, < 0.9 for shorts
6. ATR(14) trailing stoploss at 2.5x (tighter for 15m noise)
7. COOLDOWN period: 20 bars after exit before re-entry (reduces fee churn)

Key differences from failed #289:
- Stricter HTF filter: BOTH 4h AND 1h must align (not just 4h)
- Higher ADX threshold: 20 vs 15 (fewer but higher quality trades)
- Volume confirmation added (filters false breakouts)
- Cooldown period after exits (prevents rapid re-entry whipsaws)
- Conservative position size: 0.20 base, 0.30 strong trend

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_ema_crossover_dual_htf_hma_volume_atr_v1"
timeframe = "15m"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume ratio: taker_buy_volume / total_volume
    volume_ratio = np.divide(taker_buy_volume, volume, out=np.zeros_like(close), where=volume != 0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20  # Conservative base for 15m (reduce fee drag)
    SIZE_STRONG = 0.30  # Increased size in strong trend
    
    # Track position state for stoploss and cooldown
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    cooldown_counter = 0
    COOLDOWN_BARS = 20  # Wait 20 bars (5 hours) after exit before re-entry
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (Dual HTF Filter - STRICT) ===
        # 4h HMA = PRIMARY directional bias (proven edge from #292)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h HMA = SECONDARY confirmation (stricter than past 15m attempts)
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # BOTH HTF must align for trade (key difference from failed #289)
        bull_htf_aligned = bull_trend_4h and bull_trend_1h
        bear_htf_aligned = bear_trend_4h and bear_trend_1h
        
        # === TREND STRENGTH ===
        # ADX > 20 = trending market (higher threshold to reduce whipsaws)
        trending = adx[i] > 20
        strong_trend = adx[i] > 30
        
        # === EMA CROSSOVER ===
        # Fast EMA crosses above slow EMA = bullish
        ema_bullish = ema_fast[i] > ema_slow[i]
        # Fast EMA crosses below slow EMA = bearish
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # === VOLUME CONFIRMATION ===
        # Long: volume_ratio > 1.1 (buying pressure)
        # Short: volume_ratio < 0.9 (selling pressure)
        volume_bullish = volume_ratio[i] > 1.1
        volume_bearish = volume_ratio[i] < 0.9
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size
        if high_volatility:
            position_size = SIZE_BASE
        elif strong_trend:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === COOLDOWN CHECK ===
        if cooldown_counter > 0:
            cooldown_counter -= 1
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS (STRICT - all must align) ===
        new_signal = 0.0
        
        # LONG: 4h HMA up + 1h HMA up + EMA bullish + ADX trending + Volume confirmation
        long_conditions = (
            bull_htf_aligned and  # BOTH 4h and 1h bullish
            ema_bullish and  # EMA crossover bullish
            trending and  # ADX confirms trend
            volume_bullish  # Volume confirms buying
        )
        
        # SHORT: Mirror of long
        short_conditions = (
            bear_htf_aligned and  # BOTH 4h and 1h bearish
            ema_bearish and  # EMA crossover bearish
            trending and  # ADX confirms trend
            volume_bearish  # Volume confirms selling
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss hit
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss hit
        
        # === HTF TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and not bull_htf_aligned:
                new_signal = 0.0  # HTF trend reversed against long
            if position_side < 0 and not bear_htf_aligned:
                new_signal = 0.0  # HTF trend reversed against short
        
        # === EMA REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_bearish:
                new_signal = 0.0  # EMA crossed against long
            if position_side < 0 and ema_bullish:
                new_signal = 0.0  # EMA crossed against short
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                cooldown_counter = COOLDOWN_BARS  # Start cooldown after exit
        
        signals[i] = new_signal
    
    return signals