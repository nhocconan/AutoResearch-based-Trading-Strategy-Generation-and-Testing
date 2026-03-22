#!/usr/bin/env python3
"""
Experiment #010: 4h RSI Mean Reversion with 1D/1W HMA Trend Bias

Hypothesis: After 9 experiments, the pattern shows that pure trend strategies fail in 
bear/range markets (2022 crash, 2025 bear), while pure mean reversion fails in strong 
trends. This 4h strategy combines both modes with HTF regime detection:

1. 1D HMA trend bias: Determines primary direction. Price > 1d_HMA = bull bias (prefer longs),
   Price < 1d_HMA = bear bias (prefer shorts). Stable filter that changes rarely.

2. 1W HMA mega-trend: Ultra-long-term context. When price > 1w_HMA, market is in structural 
   bull. When price < 1w_HMA, structural bear. Used to adjust position sizing.

3. RSI(14) mean reversion: Long when RSI < 30 (oversold) + bull bias. Short when RSI > 70 
   (overbought) + bear bias. Proven edge in range markets.

4. ADX(14) regime filter: ADX < 20 = range (use mean reversion). ADX > 25 = trend (use 
   breakout logic). Prevents mean reversion in strong trends.

5. Bollinger Band confirmation: Long when price < BB_lower + RSI < 30. Short when price > 
   BB_upper + RSI > 70. Double confirmation reduces false signals.

6. ATR-based stoploss: 2.5 * ATR(14) trailing stop. Exits when trend reverses.

7. Asymmetric sizing: In structural bull (price > 1w_HMA), long size = 0.30, short size = 0.15.
   In structural bear (price < 1w_HMA), short size = 0.30, long size = 0.15.

Why 4h should work:
- 4h is sweet spot: fewer trades than 1h (less fee drag), more signals than 12h
- RSI mean reversion works well on 4h timeframe (proven in literature)
- Dual HTF (1d + 1w) provides stable trend context without overfitting
- Asymmetric sizing protects in bear markets while capturing bull rallies
- Target 40-70 trades/year = optimal for 4h (Rule 10)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.15-0.30 discrete, asymmetric by regime
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_mr_1d_1w_hma_adx_bbw_asym_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_BULL = 0.30  # Preferred direction in bull
    BASE_SIZE_BEAR = 0.15  # Counter-trend in bull
    BASE_SIZE_SHORT_BULL = 0.15  # Counter-trend in bull
    BASE_SIZE_SHORT_BEAR = 0.30  # Preferred direction in bear
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D HMA TREND BIAS (primary direction filter) ===
        bull_bias_1d = close[i] > hma_1d_aligned[i]
        bear_bias_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W HMA MEGA-TREND (structural context) ===
        structural_bull = close[i] > hma_1w_aligned[i]
        structural_bear = close[i] < hma_1w_aligned[i]
        
        # === ADX REGIME DETECTION ===
        is_ranging = adx_14[i] < 20  # Range market - prefer mean reversion
        is_trending = adx_14[i] > 25  # Trend market - prefer trend following
        
        # === RSI EXTREMES (mean reversion signals) ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === BOLLINGER BAND CONFIRMATION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === DETERMINE POSITION SIZING BASED ON REGIME ===
        if structural_bull:
            long_size = BASE_SIZE_BULL
            short_size = BASE_SIZE_SHORT_BULL
        else:  # structural_bear
            long_size = BASE_SIZE_BEAR
            short_size = BASE_SIZE_SHORT_BEAR
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: MEAN REVERSION (ranging market + RSI extreme + BB confirmation)
        if is_ranging:
            # Long: oversold + below BB + 1d bull bias (or neutral)
            if rsi_oversold and price_below_bb:
                if bull_bias_1d:
                    new_signal = long_size
                elif not bear_bias_1d:  # neutral
                    new_signal = long_size * 0.7
            
            # Short: overbought + above BB + 1d bear bias (or neutral)
            elif rsi_overbought and price_above_bb:
                if bear_bias_1d:
                    new_signal = -short_size
                elif not bull_bias_1d:  # neutral
                    new_signal = -short_size * 0.7
        
        # MODE 2: TREND CONTINUATION (trending market + pullback to mean)
        elif is_trending:
            # Long in uptrend: RSI pullback to 40-50 + 1d bull bias
            if bull_bias_1d and 35 < rsi_14[i] < 50:
                new_signal = long_size
            
            # Short in downtrend: RSI pullback to 50-65 + 1d bear bias
            elif bear_bias_1d and 50 < rsi_14[i] < 65:
                new_signal = -short_size
        
        # MODE 3: STRONG REVERSAL (extreme RSI + structural bias change)
        # This catches major trend changes
        if rsi_14[i] < 20 and structural_bull and bear_bias_1d:
            # Deep oversold in structural bull but 1d turned bear = potential bottom
            new_signal = long_size * 0.5  # Cautious entry
        
        if rsi_14[i] > 80 and structural_bear and bull_bias_1d:
            # Deep overbought in structural bear but 1d turned bull = potential top
            new_signal = -short_size * 0.5  # Cautious entry
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns strongly bearish
            if position_side > 0 and bear_bias_1d and adx_14[i] > 25:
                trend_reversal = True
            # Exit short if 1d bias turns strongly bullish
            if position_side < 0 and bull_bias_1d and adx_14[i] > 25:
                trend_reversal = True
        
        # === RSI EXIT (take profit on mean reversion) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long: exit when RSI recovers to 50-60
            if position_side > 0 and rsi_14[i] > 55:
                rsi_exit = True
            # Short: exit when RSI drops to 40-50
            if position_side < 0 and rsi_14[i] < 45:
                rsi_exit = True
        
        # Apply stoploss or exit signals
        if stoploss_triggered or trend_reversal or rsi_exit:
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