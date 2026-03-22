#!/usr/bin/env python3
"""
Experiment #093: 1h Fisher Transform Reversals with 4h HMA Trend + Choppiness Regime
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025+).
Combined with 4h HMA trend bias and Choppiness Index regime filter, this adapts to
market conditions: mean-revert in ranges, trend-follow in trends.

Why this might work on 1h:
- Fisher Transform normalizes price to Gaussian distribution, extreme values signal reversals
- 4h HMA provides stable trend bias without lag of daily
- Choppiness Index (CHOP) detects range vs trend regime (CHOP>61.8=range, <38.2=trend)
- Different entry logic per regime ensures trades in all market conditions
- 1h captures intraday moves while HTF filter avoids noise

Learning from failures:
- #087 (1h regime adaptive): Sharpe=-0.430 - regime logic was too complex
- #081 (1h Supertrend): Sharpe=-2.382 - Supertrend whipsawed on 1h
- Key: Fisher Transform has better reversal signals than RSI for 1h timeframe
- Keep entry conditions LOOSE to ensure trades on all symbols

Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_chop_regime_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normalized distribution.
    Extreme values (>1.5 or <-1.5) signal potential reversals.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        x = (hl2 - lowest) / (highest - lowest)
        
        # Clamp to avoid extreme values
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging/choppy market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # CHOP formula
        chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
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
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean reversion favored)
        # CHOP < 38.2 = trending market (trend following favored)
        # ADX > 25 = strong trend, ADX < 20 = weak/ranging
        is_range_regime = chop[i] > 55  # Slightly lower threshold to catch more ranges
        is_trend_regime = chop[i] < 45 and adx[i] > 20
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher < -1.5 and crossing up = bullish reversal
        # Fisher > +1.5 and crossing down = bearish reversal
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i]) else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i]) else False
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION ENTRIES ===
        if is_range_regime:
            # Long: Fisher oversold + 4h bullish bias or RSI oversold
            if fisher_oversold and (bull_trend_4h or rsi[i] < 35):
                if fisher_cross_up or rsi[i] < 30:
                    new_signal = SIZE_BASE
            
            # Short: Fisher overbought + 4h bearish bias or RSI overbought
            if fisher_overbought and (bear_trend_4h or rsi[i] > 65):
                if fisher_cross_down or rsi[i] > 70:
                    new_signal = -SIZE_BASE
        
        # === TREND REGIME: TREND FOLLOWING ENTRIES ===
        if is_trend_regime:
            # Long: 4h bullish + Fisher turning up from neutral
            if bull_trend_4h and fisher[i] > -1.0 and fisher_cross_up:
                if ema_bullish:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # Short: 4h bearish + Fisher turning down from neutral
            if bear_trend_4h and fisher[i] < 1.0 and fisher_cross_down:
                if ema_bearish:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # === FALLBACK: FISHER EXTREME REVERSALS (ensures trades) ===
        # Very extreme Fisher values signal strong reversals regardless of regime
        if new_signal == 0.0:
            # Extreme oversold = long
            if fisher[i] < -1.8 and fisher_cross_up:
                new_signal = SIZE_BASE
            
            # Extreme overbought = short
            if fisher[i] > 1.8 and fisher_cross_down:
                new_signal = -SIZE_BASE
        
        # === Additional fallback: Simple 4h trend + EMA alignment ===
        # This ensures we generate trades even when Fisher is neutral
        if new_signal == 0.0:
            if bull_trend_4h and ema_bullish and adx[i] > 18:
                # Check Fisher is not against us
                if fisher[i] > -0.5:
                    new_signal = SIZE_BASE
            
            if bear_trend_4h and ema_bearish and adx[i] > 18:
                # Check Fisher is not against us
                if fisher[i] < 0.5:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals