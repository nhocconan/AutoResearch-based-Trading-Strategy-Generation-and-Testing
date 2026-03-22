#!/usr/bin/env python3
"""
Experiment #011: 4h Supertrend with 1d Trend Filter + RSI Confirmation

Hypothesis: Previous mean-reversion strategies failed because BTC/ETH 2025 is 
bearish/trending, not ranging. Supertrend is a proven trend-following indicator 
that works well in crypto with built-in ATR stops. This strategy uses:

1. Supertrend(10, 3.0) on 4h - primary trend signal with ATR-based stops
2. 1d HMA(21) - major trend filter (only long if price > 1d HMA, short if <)
3. RSI(14) - entry timing filter (avoid chasing: RSI 35-65 for entries)
4. ADX(14) - trend strength confirmation (only trade when ADX > 20)
5. Asymmetric sizing: 0.30 for trend-aligned, 0.20 for counter-trend
6. Trailing stop: signal→0 when Supertrend flips

Why this should work:
- Supertrend proven on crypto (clear signals, built-in stops)
- 1d HMA filter prevents counter-trend trades (major failure mode in 2022)
- RSI filter avoids entering at extremes (reduces whipsaw)
- ADX ensures we only trade when trend has strength
- 4h TF targets 30-60 trades/year (optimal for trend following)
- Conservative sizing (0.20-0.30) protects against crashes

Timeframe: 4h (REQUIRED for Experiment #011)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: Built into Supertrend + signal→0 on flip
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_hma_1d_rsi_adx_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return atr.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic bands
    hl2 = (high_s + low_s) / 2
    upper_band = hl2 + multiplier * pd.Series(atr)
    lower_band = hl2 - multiplier * pd.Series(atr)
    
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    # Initialize
    supertrend[0] = upper_band.iloc[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Currently in long mode
            if close[i] > supertrend[i-1]:
                # Stay long, update lower band
                supertrend[i] = max(lower_band.iloc[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Flip to short
                supertrend[i] = upper_band.iloc[i]
                direction[i] = -1
        else:
            # Currently in short mode
            if close[i] < supertrend[i-1]:
                # Stay short, update upper band
                supertrend[i] = min(upper_band.iloc[i], supertrend[i-1])
                direction[i] = -1
            else:
                # Flip to long
                supertrend[i] = lower_band.iloc[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
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
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Additional 4h trend confirmation
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    TREND_SIZE = 0.30  # When all filters align
    REDUCED_SIZE = 0.20  # When some filters weak
    
    # Track position state
    in_position = False
    position_side = 0
    last_st_direction = 0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H SUPERTREND SIGNAL ===
        st_long = st_direction[i] == 1  # Price above supertrend
        st_short = st_direction[i] == -1  # Price below supertrend
        
        # === 4H HMA TREND ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trend has strength
        adx_very_strong = adx_14[i] > 25
        
        # === RSI ENTRY FILTER ===
        rsi_neutral = 35 <= rsi_14[i] <= 65  # Not overextended
        rsi_bullish = rsi_14[i] > 45  # Momentum supportive for long
        rsi_bearish = rsi_14[i] < 55  # Momentum supportive for short
        rsi_not_overbought = rsi_14[i] < 70
        rsi_not_oversold = rsi_14[i] > 30
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        long_confidence = 0
        
        # Primary: Supertrend long
        if st_long:
            long_confidence += 2
            
            # 1d trend alignment (major filter)
            if daily_bullish:
                long_confidence += 2
            else:
                long_confidence -= 1  # Counter-trend penalty
            
            # 4h HMA confirmation
            if hma_4h_bullish:
                long_confidence += 1
            
            # ADX strength
            if adx_strong:
                long_confidence += 1
            if adx_very_strong:
                long_confidence += 0.5
            
            # RSI filter (not chasing)
            if rsi_not_overbought and rsi_bullish:
                long_confidence += 1
            elif rsi_neutral:
                long_confidence += 0.5
            
            # Enter if confidence >= 5
            if long_confidence >= 5:
                new_signal = TREND_SIZE
            elif long_confidence >= 4:
                new_signal = REDUCED_SIZE
        
        # SHORT ENTRY
        short_confidence = 0
        
        # Primary: Supertrend short
        if st_short:
            short_confidence += 2
            
            # 1d trend alignment (major filter)
            if daily_bearish:
                short_confidence += 2
            else:
                short_confidence -= 1  # Counter-trend penalty
            
            # 4h HMA confirmation
            if hma_4h_bearish:
                short_confidence += 1
            
            # ADX strength
            if adx_strong:
                short_confidence += 1
            if adx_very_strong:
                short_confidence += 0.5
            
            # RSI filter (not chasing)
            if rsi_not_oversold and rsi_bearish:
                short_confidence += 1
            elif rsi_neutral:
                short_confidence += 0.5
            
            # Enter if confidence >= 5
            if short_confidence >= 5:
                new_signal = -TREND_SIZE
            elif short_confidence >= 4:
                new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~10 days on 4h), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if st_long and daily_bullish and rsi_not_overbought:
                new_signal = REDUCED_SIZE
            elif st_short and daily_bearish and rsi_not_oversold:
                new_signal = -REDUCED_SIZE
        
        # === SUPERTREND FLIP EXIT (built-in stoploss) ===
        supertrend_flip = False
        if in_position and position_side != 0:
            # Exit long if supertrend flips short
            if position_side > 0 and st_short:
                supertrend_flip = True
            # Exit short if supertrend flips long
            if position_side < 0 and st_long:
                supertrend_flip = True
        
        # === MAJOR TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI very overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short if RSI very oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply exits
        if supertrend_flip or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                last_trade_bar = i
        
        signals[i] = new_signal
        last_st_direction = st_direction[i]
    
    return signals