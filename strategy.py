#!/usr/bin/env python3
"""
Experiment #434: 30m Regime-Adaptive Strategy with 4h Trend Filter

Hypothesis: After 433 failed experiments, the key insight is that BTC/ETH need
REGIME-ADAPTIVE logic - different strategies for trending vs ranging markets.
Simple trend-following fails in 2022 crash and 2025 bear market.

This strategy uses:
1. 4h HMA(21) for trend bias (via mtf_data helper - call ONCE before loop)
2. Choppiness Index(14) for regime detection:
   - CHOP > 61.8 = ranging (mean reversion at BB bounds)
   - CHOP < 38.2 = trending (pullback entries with trend)
3. RSI(14) + Bollinger(20, 2.0) for mean reversion in ranges
4. Fisher Transform(9) for reversal confirmation in trends
5. ATR(14) trailing stop at 2.5x for risk management

Why 30m:
- More trades than 1h/4h (ensures >10 trades/year)
- Less noise than 5m/15m (better signal quality)
- Works well with 4h HTF filter (8x ratio is optimal)

Position sizing: 0.30 discrete (conservative for 30m volatility)
Stoploss: 2.5 * ATR(14) trailing
Target: Beat Sharpe=0.676 from current best (4h strategy)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_chop_4h_hma_fisher_bb_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate median price
        median = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize price to -1 to +1 range
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            continue
        
        normalized = 2.0 * ((median - lowest) / (highest - lowest)) - 1.0
        
        # Apply Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        fisher[i] = fisher_val
        
        # Trigger line (1-period lag of fisher)
        if i > period - 1:
            trigger[i] = fisher[i - 1]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx_arr = np.zeros(n)
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx_arr[i] = 100 * np.abs(plus_di - minus_di) / di_sum
        
        if i == period:
            adx[i] = dx_arr[i]
        elif not np.isnan(adx[i-1]):
            adx[i] = ((adx[i-1] * (period - 1)) + dx_arr[i]) / period
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        neutral_market = not ranging_market and not trending_market
        
        # === MEAN REVERSION SIGNALS (for ranging market) ===
        # Long: Price at lower BB + RSI oversold
        mr_long = (close[i] <= bb_lower[i] * 1.002) and (rsi[i] < 40)
        # Short: Price at upper BB + RSI overbought
        mr_short = (close[i] >= bb_upper[i] * 0.998) and (rsi[i] > 60)
        
        # === TREND PULLBACK SIGNALS (for trending market) ===
        # Long: Bullish trend + RSI pullback + Fisher confirmation
        trend_long = (bull_trend_4h and 
                      rsi[i] < 45 and 
                      rsi[i-1] >= 45 and  # RSI just crossed below 45
                      fisher[i] > -1.5 and 
                      (i > 0 and (np.isnan(fisher_trigger[i]) or fisher[i] > fisher_trigger[i])))
        
        # Short: Bearish trend + RSI pullback + Fisher confirmation
        trend_short = (bear_trend_4h and 
                       rsi[i] > 55 and 
                       rsi[i-1] <= 55 and  # RSI just crossed above 55
                       fisher[i] < 1.5 and 
                       (i > 0 and (np.isnan(fisher_trigger[i]) or fisher[i] < fisher_trigger[i])))
        
        # === NEUTRAL MARKET (use ADX filter) ===
        # If ADX > 25, treat as trending; if < 20, treat as ranging
        if neutral_market:
            if adx[i] > 25:
                trending_market = True
                ranging_market = False
            elif adx[i] < 20:
                ranging_market = True
                trending_market = False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Mean reversion entries (ranging market)
        if ranging_market:
            if mr_long:
                new_signal = SIZE
            elif mr_short:
                new_signal = -SIZE
        
        # Trend pullback entries (trending market)
        if trending_market and new_signal == 0.0:
            if trend_long:
                new_signal = SIZE
            elif trend_short:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
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
        
        signals[i] = new_signal
    
    return signals