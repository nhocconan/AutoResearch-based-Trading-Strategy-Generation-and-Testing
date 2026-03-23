#!/usr/bin/env python3
"""
Experiment #019: 4h Primary + 1d HTF — Fisher Transform + Vol Spike Reversion + ADX Regime

Hypothesis: Combining Ehlers Fisher Transform (reversal detection) with volatility spike
reversion and ADX regime filtering will work better in bear/range markets (2025) while
still capturing trends. Fisher Transform normalizes price to -1.5 to +1.5 range, making
extremes clear for reversals. Vol spike (ATR7/ATR30 > 1.8) indicates panic bottoms/tops.

Key components:
1. Fisher Transform (period=9): Normalized oscillator for reversal detection
2. Volatility Spike: ATR(7)/ATR(30) ratio > 1.8 = extreme vol = reversion likely
3. ADX(14): Regime filter (ADX>25 trend, ADX<20 range)
4. 1d HMA: Macro trend bias (only trade with daily trend for higher win rate)
5. Bollinger Bands: Additional mean reversion confirmation
6. ATR trailing stop: 2.5*ATR for risk management

Why this should work:
- Fisher Transform catches reversals better than RSI in bear markets
- Vol spike filter ensures we enter at panic extremes (high reward/risk)
- ADX regime adapts between trend-follow and mean-revert modes
- 4h primary = 20-50 trades/year target (fee-efficient)
- 1d HTF = strong trend filter, avoids counter-trend trades

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volspike_adx_regime_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    """
    n = len(close)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Normalize typical price to -1 to +1 range
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest + 1e-10
    normalized = 2.0 * (typical - lowest) / price_range - 1.0
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher Transform
    fisher = np.zeros(n)
    fisher_line = np.zeros(n)
    
    for i in range(period, n):
        if np.abs(1.0 - normalized[i]) < 1e-10 or np.abs(1.0 + normalized[i]) < 1e-10:
            fisher_line[i] = fisher_line[i-1]
        else:
            fisher_line[i] = 0.66 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i])) + 0.67 * fisher_line[i-1]
        fisher[i] = fisher_line[i]
    
    return fisher, fisher_line

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
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_line = calculate_fisher_transform(high, low, close, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volatility spike ratio
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(vol_ratio[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE (panic extreme) ===
        vol_spike = vol_ratio[i] > 1.8  # ATR7 > 1.8 * ATR30 = extreme vol
        
        # === FISHER TRANSFORM EXTREMES ===
        fisher_oversold = fisher[i] < -1.5  # Strong reversal signal
        fisher_overbought = fisher[i] > 1.5  # Strong reversal signal
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === ADX REGIME ===
        adx_trending = adx[i] > 25.0
        adx_ranging = adx[i] < 20.0
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DI CROSSOVER ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- VOL SPIKE REVERSION (high priority - panic extremes) ---
        if vol_spike:
            # Long: Vol spike + Fisher oversold + price below BB + daily bias helps
            if fisher_oversold and price_below_bb_lower:
                if price_above_hma_1d or rsi_oversold:  # Either daily bullish OR RSI extreme
                    new_signal = POSITION_SIZE
            
            # Short: Vol spike + Fisher overbought + price above BB + daily bias helps
            elif fisher_overbought and price_above_bb_upper:
                if price_below_hma_1d or rsi_overbought:  # Either daily bearish OR RSI extreme
                    new_signal = -POSITION_SIZE
        
        # --- ADX TRENDING REGIME ---
        elif adx_trending:
            # Long: DI bullish + Fisher rising from oversold + daily confirms
            if di_bullish and fisher_rising:
                if fisher[i] > -1.0 and price_above_hma_1d:  # Fisher recovering + daily bullish
                    new_signal = POSITION_SIZE
            
            # Short: DI bearish + Fisher falling from overbought + daily confirms
            elif di_bearish and fisher_falling:
                if fisher[i] < 1.0 and price_below_hma_1d:  # Fisher dropping + daily bearish
                    new_signal = -POSITION_SIZE
        
        # --- ADX RANGING REGIME (mean reversion) ---
        elif adx_ranging:
            # Long: Fisher oversold + price below BB (double mean reversion)
            if fisher_oversold and price_below_bb_lower:
                new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + price above BB (double mean reversion)
            elif fisher_overbought and price_above_bb_upper:
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
        
        # === EXIT ON STRONG REGIME CHANGE ===
        # Exit long if daily trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and fisher_falling and fisher[i] < 0:
                new_signal = 0.0
        
        # Exit short if daily trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and fisher_rising and fisher[i] > 0:
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